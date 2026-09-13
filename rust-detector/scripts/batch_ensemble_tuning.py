#!/usr/bin/env python3
"""Tune and review the ensemble detector across a directory of country series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    labels: Path
    rust_binary: Path
    work_directory: Path
    evaluations: int
    population_size: int
    proximity_days: int
    coalesce_sparse_gap_days: int
    threads_per_tuning: int
    seed: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-constrained ensemble tuning across country series."
    )
    parser.add_argument("--series-directory", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--work-directory", required=True, type=Path)
    parser.add_argument("--evaluations", type=int, default=5_000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--proximity-days", type=int, default=3)
    parser.add_argument(
        "--coalesce-sparse-gap-days",
        type=int,
        default=1,
        help="Maximum normal-day gap for combining sparse anomalies.",
    )
    parser.add_argument(
        "--multi-country-threshold",
        type=int,
        default=5,
        help="Distinct countries required to group same-day candidates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-tuning", type=int, default=1)
    parser.add_argument("--countries", nargs="+", help="Optional country-code subset.")
    parser.add_argument(
        "--dataset-description",
        default="public cenalert-paper <Virtual private network> topic series (2011–2024)",
        help="Dataset description written to the report.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-diagnostics",
        action="store_true",
        help="Re-evaluate saved infeasible Pareto fronts when resuming.",
    )
    arguments = parser.parse_args()
    if arguments.evaluations < arguments.population_size:
        parser.error("--evaluations must be at least --population-size")
    if arguments.population_size < 2 or arguments.population_size % 2:
        parser.error("--population-size must be an even integer of at least 2")
    if arguments.proximity_days < 0:
        parser.error("--proximity-days must be non-negative")
    if arguments.coalesce_sparse_gap_days < 0:
        parser.error("--coalesce-sparse-gap-days must be non-negative")
    if arguments.multi_country_threshold < 2:
        parser.error("--multi-country-threshold must be at least 2")
    if arguments.workers < 1 or arguments.threads_per_tuning < 1:
        parser.error("worker and thread counts must be positive")
    return arguments


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or f"command exited with {completed.returncode}")
    return completed


def country_seed(base_seed: int, country: str) -> int:
    return base_seed + sum(
        (index + 1) * ord(character) for index, character in enumerate(country)
    )


def source_country(source: Path) -> str:
    if source.suffix.lower() == ".csv":
        return source.stem.upper()
    records = json.loads(source.read_text())
    countries = {
        str(record.get("country", "")).strip().upper()
        for record in records
        if record.get("country")
    }
    if len(countries) != 1:
        raise ValueError(f"{source} must contain exactly one country")
    return countries.pop()


def convert_series(source: Path, destination: Path, country: str) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    zero_count = 0
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as input_file:
            records = list(csv.DictReader(input_file))
        default_topic = "012t0g"
    else:
        records = json.loads(source.read_text())
        default_topic = "vpn"
    for row in records:
        value = float(row.get("rate", row.get("value")))
        zero_count += math.isclose(value, 0.0, abs_tol=1e-12)
        observations.append(
            {
                "country": country,
                "date": row["date"][:10],
                "rate": value,
                "topic": row.get("topic", default_topic),
            }
        )
    if not observations:
        raise ValueError(f"{source} contains no observations")
    observations.sort(key=lambda observation: observation["date"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(observations), encoding="utf-8")
    return {
        "observations": len(observations),
        "zero_observations": zero_count,
        "first_date": observations[0]["date"],
        "last_date": observations[-1]["date"],
    }


def label_applies(label: dict[str, Any], country: str) -> bool:
    if label["scope"] == "country":
        code = label.get("country_code")
        additional = [
            code.upper() for code in label.get("additional_matching_countries", [])
        ]
        return code is None or code.upper() == country or country in additional
    affected = [code.upper() for code in label.get("affected_countries", [])]
    return not affected or country in affected


def label_matches(label: dict[str, Any], anomaly_date: date, proximity_days: int) -> bool:
    distance = (anomaly_date - date.fromisoformat(label["date"])).days
    if label.get("match_mode", "proximity") == "exact":
        return distance == 0
    return abs(distance) <= proximity_days


def required_labels(
    labels: list[dict[str, Any]],
    country: str,
    first_date: str,
    last_date: str,
    proximity_days: int,
) -> list[dict[str, Any]]:
    first = date.fromisoformat(first_date)
    last = date.fromisoformat(last_date)
    return [
        label
        for label in labels
        if label["scope"] == "country"
        and label.get("match_mode", "proximity") == "proximity"
        and (
            label.get("country_code") is None
            or label["country_code"].upper() == country
        )
        and (date.fromisoformat(label["date"]) - first).days >= -proximity_days
        and (date.fromisoformat(label["date"]) - last).days <= proximity_days
    ]


def matched_label(
    anomaly: dict[str, Any],
    labels: list[dict[str, Any]],
    country: str,
    proximity_days: int,
) -> dict[str, Any] | None:
    anomaly_date = date.fromisoformat(anomaly["start"])
    candidates = [
        (label, (anomaly_date - date.fromisoformat(label["date"])).days, index)
        for index, label in enumerate(labels)
        if label_applies(label, country)
        and label_matches(label, anomaly_date, proximity_days)
    ]
    if not candidates:
        return None
    label, distance, _ = min(
        candidates,
        key=lambda candidate: (
            abs(candidate[1]),
            candidate[0]["scope"] == "global",
            candidate[0]["date"],
            candidate[2],
        ),
    )
    return {"label": label, "distance_days": distance}


def coverage(
    labels: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    proximity_days: int,
) -> int:
    return sum(
        any(
            label_matches(
                label, date.fromisoformat(anomaly["start"]), proximity_days
            )
            for anomaly in anomalies
        )
        for label in labels
    )


def ensemble_parameters(values: list[float]) -> dict[str, float | int]:
    return {
        "window": round(values[0]),
        "min_residual_ratio": values[1],
        "efficiency": values[2],
        "chebyshev_z": values[3],
        "chebyshev_k": values[4],
        "isolation_forest_min_score": values[5],
        "lof_min_score": values[6],
    }


def detect(config: Config, input_path: Path, parameter_path: Path) -> dict[str, Any]:
    completed = run_command(
        [
            str(config.rust_binary),
            "detect",
            "--algorithm",
            "ensemble",
            "--input",
            str(input_path),
            "--parameters",
            str(parameter_path),
        ]
    )
    return json.loads(completed.stdout)


def filter_candidates(
    config: Config, detection_path: Path, output_path: Path
) -> dict[str, Any]:
    run_command(
        [
            str(config.rust_binary),
            "filter",
            "--input",
            str(detection_path),
            "--labels",
            str(config.labels),
            "--proximity-days",
            str(config.proximity_days),
            "--coalesce-sparse-gap-days",
            str(config.coalesce_sparse_gap_days),
            "--output",
            str(output_path),
        ]
    )
    return json.loads(output_path.read_text())


def tune(
    config: Config,
    input_path: Path,
    output_path: Path,
    country: str,
    constrained: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(config.rust_binary),
        "tune",
        "--algorithm",
        "ensemble",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--evaluations",
        str(config.evaluations),
        "--population-size",
        str(config.population_size),
        "--proximity-days",
        str(config.proximity_days),
        "--seed",
        str(country_seed(config.seed, country)),
        "--threads",
        str(config.threads_per_tuning),
    ]
    if constrained:
        command.extend(["--labels", str(config.labels)])
    else:
        proximity_index = command.index("--proximity-days")
        del command[proximity_index : proximity_index + 2]
    return run_command(command)


def select(
    config: Config, front_path: Path, parameter_path: Path, left_position: int | None = None
) -> subprocess.CompletedProcess[str]:
    command = [
        str(config.rust_binary),
        "select",
        "--algorithm",
        "ensemble",
        "--input",
        str(front_path),
        "--output",
        str(parameter_path),
    ]
    if left_position is not None:
        command.extend(["--left-position", str(left_position)])
    return run_command(command)


def diagnose_infeasible(
    config: Config,
    country: str,
    input_path: Path,
    required: list[dict[str, Any]],
    directory: Path,
) -> dict[str, Any]:
    front_path = directory / "unconstrained-front.json"
    if not front_path.exists():
        tune(config, input_path, front_path, country, constrained=False)
    front = json.loads(front_path.read_text())
    best: tuple[int, int, float, list[dict[str, Any]]] | None = None
    labels_seen = [False] * len(required)
    for index, (values, objectives) in enumerate(front):
        parameter_path = directory / f"diagnostic-{index}.json"
        parameter_path.write_text(json.dumps(ensemble_parameters(values)))
        anomalies = detect(config, input_path, parameter_path)["anomalies"]
        for label_index, label in enumerate(required):
            labels_seen[label_index] |= any(
                label_matches(
                    label,
                    date.fromisoformat(anomaly["start"]),
                    config.proximity_days,
                )
                for anomaly in anomalies
            )
        candidate = (
            coverage(required, anomalies, config.proximity_days),
            -int(objectives[0]),
            -float(objectives[1]),
            anomalies,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return {
            "best_simultaneous_labels_covered": 0,
            "best_solution_unpreserved_labels": required,
            "labels_never_detected": required,
        }
    best_unpreserved = [
        label
        for label in required
        if not any(
            label_matches(label, date.fromisoformat(anomaly["start"]), config.proximity_days)
            for anomaly in best[3]
        )
    ]
    return {
        "best_simultaneous_labels_covered": best[0],
        "best_solution_unpreserved_labels": best_unpreserved,
        "labels_never_detected": [
            label for label, seen in zip(required, labels_seen) if not seen
        ],
    }


def run_country(
    source: Path,
    country: str,
    config: Config,
    resume: bool,
    refresh_diagnostics: bool,
) -> dict[str, Any]:
    directory = config.work_directory / country
    result_path = directory / "result.json"
    if resume and result_path.exists():
        saved = json.loads(result_path.read_text())
        if not refresh_diagnostics or saved["status"] != "no_feasible_solution":
            return saved
        labels = json.loads(config.labels.read_text())
        required = required_labels(
            labels,
            country,
            saved["first_date"],
            saved["last_date"],
            config.proximity_days,
        )
        saved.update(
            diagnose_infeasible(
                config, country, directory / "series.json", required, directory
            )
        )
        saved.pop("maximum_labels_covered", None)
        saved.pop("unpreserved_labels", None)
        result_path.write_text(json.dumps(saved, indent=2) + "\n")
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {"country": country}
    try:
        input_path = directory / "series.json"
        details = convert_series(source, input_path, country)
        result.update(details)
        if details["zero_observations"] == details["observations"]:
            result.update(status="skipped", detail="all-zero series")
            return result

        labels = json.loads(config.labels.read_text())
        required = required_labels(
            labels,
            country,
            details["first_date"],
            details["last_date"],
            config.proximity_days,
        )
        result["required_labels"] = len(required)
        front_path = directory / "pareto-front.json"
        try:
            if not (resume and front_path.exists()):
                tune(config, input_path, front_path, country, constrained=True)
        except RuntimeError as error:
            if "detects every label" not in str(error):
                raise
            diagnosis = diagnose_infeasible(config, country, input_path, required, directory)
            result.update(status="no_feasible_solution", detail=str(error), **diagnosis)
            return result

        front = json.loads(front_path.read_text())
        knee_parameters = directory / "knee-parameters.json"
        left_parameters = directory / "leftmost-parameters.json"
        knee_output_path = directory / "knee-output.json"
        left_output_path = directory / "leftmost-output.json"
        knee_selection = select(config, front_path, knee_parameters)
        select(config, front_path, left_parameters, left_position=1)
        if resume and knee_output_path.exists() and left_output_path.exists():
            knee_output = json.loads(knee_output_path.read_text())
            left_output = json.loads(left_output_path.read_text())
        else:
            knee_output = detect(config, input_path, knee_parameters)
            left_output = detect(config, input_path, left_parameters)
            knee_output_path.write_text(json.dumps(knee_output, indent=2) + "\n")
            left_output_path.write_text(json.dumps(left_output, indent=2) + "\n")
        filtered = filter_candidates(
            config, knee_output_path, directory / "notification-candidates.json"
        )
        knee_anomalies = knee_output["anomalies"]
        left_anomalies = left_output["anomalies"]
        knee_found = not knee_selection.stderr.startswith("no knee found")
        method_match = re.search(r"using ([A-Za-z]+)", knee_selection.stderr)
        applicable = [label for label in labels if label_applies(label, country)]
        unmatched = [
            anomaly
            for anomaly in knee_anomalies
            if matched_label(anomaly, applicable, country, config.proximity_days) is None
        ]
        result.update(
            status="completed",
            pareto_solutions=len(front),
            knee_found=knee_found,
            selection_method=method_match.group(1) if method_match else "leftmost_fallback",
            knee_anomalies=len(knee_anomalies),
            leftmost_anomalies=len(left_anomalies),
            knee_equals_leftmost_parameters=json.loads(knee_parameters.read_text())
            == json.loads(left_parameters.read_text()),
            knee_equals_leftmost_anomalies=knee_anomalies == left_anomalies,
            required_labels_covered=coverage(required, knee_anomalies, config.proximity_days),
            anomalies_with_applicable_label=len(knee_anomalies) - len(unmatched),
            unmatched_anomalies=unmatched,
            total_impact=sum(anomaly["impact"] for anomaly in knee_anomalies),
            notification_candidates=len(filtered["anomalies"]),
            filtered_anomalies=len(filtered["discarded"]),
            sparse_anomalies_absorbed=filtered.get("coalescing", {}).get(
                "absorbed_anomalies", 0
            ),
            minimum_labeled_impact=filtered["stages"][0].get("impact_threshold"),
        )
        return result
    except Exception as error:  # Keep the country batch running for review.
        result.update(status="error", detail=f"{type(error).__name__}: {error}")
        return result
    finally:
        result["elapsed_seconds"] = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        result_path.write_text(json.dumps(result, indent=2) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(arguments: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result["status"] != "completed":
            continue
        country_directory = arguments.work_directory / result["country"]
        result["selected_parameters"] = json.loads(
            (country_directory / "knee-parameters.json").read_text()
        )
        result["leftmost_parameters"] = json.loads(
            (country_directory / "leftmost-parameters.json").read_text()
        )
    unmatched_rows = []
    candidate_rows = []
    unpreserved_rows = []
    for result in results:
        for anomaly in result.get("unmatched_anomalies", []):
            unmatched_rows.append({"country": result["country"], **anomaly})
        if result["status"] == "completed":
            candidate_path = (
                arguments.work_directory
                / result["country"]
                / "notification-candidates.json"
            )
            if candidate_path.exists():
                candidates = json.loads(candidate_path.read_text())["anomalies"]
                candidate_rows.extend(
                    {"country": result["country"], **anomaly} for anomaly in candidates
                )
        never_detected = {
            (label["date"], label.get("description", ""))
            for label in result.get("labels_never_detected", [])
        }
        for label in result.get("best_solution_unpreserved_labels", []):
            unpreserved_rows.append(
                {
                    "country": result["country"],
                    "date": label["date"],
                    "description": label.get("description", ""),
                    "category": label.get("metadata", {}).get("category", ""),
                    "verification": label.get("metadata", {}).get("verification", ""),
                    "source": label.get("metadata", {}).get("source", ""),
                    "seen_in_any_diagnostic_solution": (
                        label["date"], label.get("description", "")
                    )
                    not in never_detected,
                }
            )
    unmatched_rows.sort(key=lambda row: float(row["impact"]), reverse=True)
    write_csv(
        output / "ensemble-tuning-unlabeled-anomalies.csv",
        ["country", "start", "end", "peak", "score", "residual", "impact"],
        unmatched_rows,
    )
    candidate_rows.sort(key=lambda row: (row["country"], row["start"]))
    write_csv(
        output / "ensemble-notification-candidates.csv",
        [
            "country",
            "start",
            "end",
            "peak",
            "peak_value",
            "score",
            "residual",
            "impact",
            "sparse",
            "component_starts",
        ],
        candidate_rows,
    )
    candidates_json = output / "ensemble-notification-candidates.json"
    candidates_json.write_text(json.dumps(candidate_rows, indent=2) + "\n")
    grouped_output = output / "cross-country-notifications.json"
    run_command(
        [
            str(arguments.rust_binary.resolve()),
            "group-cross-country",
            "--input",
            str(candidates_json.resolve()),
            "--minimum-countries",
            str(arguments.multi_country_threshold),
            "--output",
            str(grouped_output.resolve()),
        ]
    )
    cross_country = json.loads(grouped_output.read_text())
    occurrence_rows = [
        {
            **occurrence,
            "countries": ",".join(occurrence["countries"]),
        }
        for occurrence in cross_country["same_day_occurrences"]
    ]
    write_csv(
        output / "cross-country-occurrences.csv",
        [
            "date",
            "country_count",
            "countries",
            "anomaly_count",
            "total_impact",
            "grouped_as_notification",
        ],
        occurrence_rows,
    )
    selected_parameters = {
        result["country"]: {
            "selected": result["selected_parameters"],
            "leftmost": result["leftmost_parameters"],
        }
        for result in results
        if result["status"] == "completed"
    }
    (output / "ensemble-tuning-selected-parameters.json").write_text(
        json.dumps(selected_parameters, indent=2) + "\n"
    )
    write_report(arguments, results, unmatched_rows, unpreserved_rows, cross_country)


def write_report(
    arguments: argparse.Namespace,
    results: list[dict[str, Any]],
    unmatched: list[dict[str, Any]],
    unpreserved: list[dict[str, Any]],
    cross_country: dict[str, Any],
) -> None:
    completed = [result for result in results if result["status"] == "completed"]
    statuses = {
        status: sum(result["status"] == status for result in results)
        for status in ("completed", "no_feasible_solution", "skipped", "error")
    }
    knee_found = sum(result.get("knee_found") is True for result in completed)
    knee_differs = sum(
        result.get("knee_equals_leftmost_anomalies") is False for result in completed
    )
    never_detected = sum(len(result.get("labels_never_detected", [])) for result in results)
    selected_anomalies = sum(result["knee_anomalies"] for result in completed)
    leftmost_anomalies = sum(result["leftmost_anomalies"] for result in completed)
    notification_candidates = sum(
        result.get("notification_candidates", 0) for result in completed
    )
    filtered_anomalies = sum(result.get("filtered_anomalies", 0) for result in completed)
    sparse_anomalies_absorbed = sum(
        result.get("sparse_anomalies_absorbed", 0) for result in completed
    )
    diagnostic_finding = (
        f"{never_detected} labels received no matches across their country's diagnostic "
        "unconstrained Pareto front and remain candidates for manual review."
        if never_detected
        else "Every reviewed required label was detected by at least one diagnostic solution."
    )
    lines = [
        "# All-country ensemble tuning report",
        "",
        f"Dataset: {arguments.dataset_description}.",
        "These results serve as diagnostics and require validation before production use.",
        "",
        "## Configuration",
        "",
        f"- Countries discovered: {len(results)}",
        f"- Unified labels supplied: {len(json.loads(arguments.labels.read_text()))}",
        f"- NSGA-II evaluations per non-zero country: {arguments.evaluations:,}",
        f"- Population size: {arguments.population_size}",
        f"- Label proximity: ±{arguments.proximity_days} days",
        "- Sparse anomaly coalescing: at most "
        f"{arguments.coalesce_sparse_gap_days} normal day(s) between components",
        "- Cross-country grouping threshold: "
        f"{arguments.multi_country_threshold} distinct countries on the same start date",
        f"- Base random seed: {arguments.seed} (deterministically offset by country code)",
        "- Constraint: every applicable proximity-mode country label must be detected",
        "- Global and exact-match pseudo-labels are attribution-only",
        "",
        "## Summary",
        "",
        f"- Completed with a feasible solution: {statuses['completed']}",
        f"- No feasible solution found: {statuses['no_feasible_solution']}",
        f"- Skipped all-zero series: {statuses['skipped']}",
        f"- Execution errors: {statuses['error']}",
        f"- Countries with an interior knee: {knee_found}",
        f"- Countries where knee and leftmost detected different anomalies: {knee_differs}",
        f"- Unmatched anomalies from selected solutions: {len(unmatched)}",
        f"- Notification candidates after impact filtering: {notification_candidates}",
        f"- Anomalies removed by impact filtering: {filtered_anomalies}",
        f"- Sparse anomalies absorbed by temporal coalescing: {sparse_anomalies_absorbed}",
        "- Dates with candidates in at least two countries: "
        f"{len(cross_country['same_day_occurrences'])}",
        "- Cross-country grouped notifications: "
        f"{cross_country['grouped_notification_count']}",
        "- Labels missed by each infeasible country's best "
        f"simultaneous-coverage solution: {len(unpreserved)}",
        f"- Of those, labels never detected by any diagnostic Pareto solution: {never_detected}",
        "",
        "## Findings",
        "",
        "The selected knee or fallback solutions detect "
        f"{selected_anomalies:,} anomalies in total,",
        f"compared with {leftmost_anomalies:,} for the first feasible Pareto solutions. Because",
        "every reported Pareto solution satisfies the country-label constraint, the leftmost",
        "solutions preserve the same required labels while detecting",
        f"{selected_anomalies - leftmost_anomalies:,} fewer additional anomalies. An interior knee",
        f"was found for {knee_found} countries, and it produced a different anomaly set from the",
        f"leftmost solution in all {knee_differs} cases. This supports evaluating a label-aware",
        "left-side selection rule. The unsupervised knee requires separate justification.",
        "",
        diagnostic_finding,
        "A different random seed,",
        "larger search, or different Google Trends series could change the result.",
        "",
        "## Labels requiring review",
        "",
        "| Country | Date | Description | Category | Source | Seen in any diagnostic solution |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label in unpreserved:
        description = str(label["description"]).replace("|", "\\|")
        lines.append(
            f"| {label['country']} | {label['date']} | {description} | "
            f"{label['category']} | {label['source']} | "
            f"{label['seen_in_any_diagnostic_solution']} |"
        )
    lines.extend(
        [
            "",
            "## Country results",
            "",
            "| Country | Status | Required labels | Covered | Knee anomalies | "
            "Leftmost anomalies | Knee found | Unmatched | Candidates | Impact threshold |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        lines.append(
            (
                "| {country} | {status} | {required_labels} | {covered} | "
                "{knee} | {left} | {knee_found} | {unmatched} | "
                "{candidates} | {threshold} |"
            ).format(
                country=result["country"],
                status=result["status"],
                required_labels=result.get("required_labels", ""),
                covered=result.get(
                    "required_labels_covered",
                    result.get("best_simultaneous_labels_covered", ""),
                ),
                knee=result.get("knee_anomalies", ""),
                left=result.get("leftmost_anomalies", ""),
                knee_found=result.get("knee_found", ""),
                unmatched=len(result.get("unmatched_anomalies", [])),
                candidates=result.get("notification_candidates", ""),
                threshold=(
                    f"{result['minimum_labeled_impact']:.3f}"
                    if result.get("minimum_labeled_impact") is not None
                    else ""
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Highest-impact unmatched anomalies",
            "",
            "The complete list is in `ensemble-tuning-unlabeled-anomalies.csv`.",
            "",
            "| Country | Start | End | Peak | Impact |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for anomaly in unmatched[:25]:
        lines.append(
            f"| {anomaly['country']} | {anomaly['start']} | {anomaly['end']} | "
            f"{anomaly['peak']} | {float(anomaly['impact']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `ensemble-tuning-unlabeled-anomalies.csv`: all selected anomalies",
            "  without an applicable label",
            "- `ensemble-tuning-selected-parameters.json`: selected and leftmost",
            "  parameters for every feasible country",
            "- `ensemble-notification-candidates.csv`: anomalies retained by the",
            "  sparse coalescing and minimum labeled-impact stages",
            "- `ensemble-notification-candidates.json`: the same candidates in",
            "  machine-readable form for notification grouping",
            "- `cross-country-notifications.json`: every date shared by at least",
            "  two countries and notifications grouped at the configured threshold",
            "- `cross-country-occurrences.csv`: review table of all dates shared by",
            "  at least two countries",
            "",
        ]
    )
    (arguments.output_directory / "ensemble-tuning-report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    selected = {country.upper() for country in arguments.countries or []}
    series = sorted(
        [
            *arguments.series_directory.glob("*.csv"),
            *arguments.series_directory.glob("*.json"),
        ]
    )
    sources = [(path, source_country(path)) for path in series]
    if len({country for _, country in sources}) != len(sources):
        raise SystemExit("series directory contains duplicate country inputs")
    if selected:
        sources = [(path, country) for path, country in sources if country in selected]
    if not sources:
        raise SystemExit("no matching country series found")
    arguments.work_directory.mkdir(parents=True, exist_ok=True)
    config = Config(
        labels=arguments.labels.resolve(),
        rust_binary=arguments.rust_binary.resolve(),
        work_directory=arguments.work_directory.resolve(),
        evaluations=arguments.evaluations,
        population_size=arguments.population_size,
        proximity_days=arguments.proximity_days,
        coalesce_sparse_gap_days=arguments.coalesce_sparse_gap_days,
        threads_per_tuning=arguments.threads_per_tuning,
        seed=arguments.seed,
    )
    results = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                run_country,
                source,
                country,
                config,
                arguments.resume,
                arguments.refresh_diagnostics,
            ): country
            for source, country in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(futures)}] {result['country']}: {result['status']}",
                flush=True,
            )
    results.sort(key=lambda result: result["country"])
    write_outputs(arguments, results)
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("completed", "no_feasible_solution", "skipped", "error")
    }
    print(json.dumps(counts, sort_keys=True))
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
