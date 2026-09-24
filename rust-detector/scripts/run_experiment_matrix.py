#!/usr/bin/env python3
"""Run the detector × label-constraint ablation on a fixed dataset."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

from batch_ensemble_tuning import (
    Config, convert_series, country_seed, detect, ensemble_parameters, label_matches,
    matched_label, required_labels, select, source_country, tune, write_csv,
)


VARIANTS = {
    "original": ("chebyshev", False),
    "labels": ("chebyshev", True),
    "ensemble": ("ensemble", False),
    "labels_ensemble": ("ensemble", True),
}
SCHEMA_VERSION = 1


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-directory", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path,
                        help="Unified labels; used for reporting in all four groups.")
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--evaluations", type=int, default=5000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--proximity-days", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-tuning", type=int, default=1)
    parser.add_argument("--selections", choices=["knee", "leftmost"], nargs="+",
                        default=["knee", "leftmost"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.population_size < 4 or args.population_size % 2:
        parser.error("--population-size must be even and at least 4")
    if args.evaluations < args.population_size:
        parser.error("--evaluations must be at least --population-size")
    if args.proximity_days < 0 or min(args.workers, args.threads_per_tuning) < 1:
        parser.error("proximity must be non-negative; worker/thread counts must be positive")
    if any(seed < 0 or seed > 2**64 - 1 - 10000 for seed in args.seeds):
        parser.error("seeds must fit unsigned 64-bit integers with room for country offsets")
    args.seeds = sorted(set(args.seeds))
    args.selections = sorted(set(args.selections))
    return args


def prepare_inputs(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    sources = sorted([*args.series_directory.glob("*.csv"),
                      *args.series_directory.glob("*.json")])
    selected = {code.upper() for code in args.countries or []}
    by_country: dict[str, Path] = {}
    for source in sources:
        country = source_country(source)
        if selected and country not in selected:
            continue
        if country in by_country:
            raise ValueError(f"duplicate country input: {country}")
        by_country[country] = source
    if selected - by_country.keys():
        raise ValueError(f"missing requested countries: {sorted(selected - by_country.keys())}")
    if not by_country:
        raise ValueError("no country series found")
    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a nonempty JSON array")
    output = args.output_directory
    manifest_path = output / "experiment.json"
    existing = manifest_path.exists()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "countries": sorted(by_country), "seeds": args.seeds,
        "evaluations": args.evaluations, "population_size": args.population_size,
        "proximity_days": args.proximity_days,
        "threads_per_tuning": args.threads_per_tuning,
        "selections": args.selections, "variants": VARIANTS,
        "k_min": 5.0, "k_max": 18.0,
        "initial_population_seeded": False,
        "efficiency_bounds": {"chebyshev": [0.01, 0.1], "ensemble": [0.05, 0.1]},
        "postprocessing": False,
    }
    # JSON normalizes tuples so manifests can be compared after deserialization.
    manifest = json.loads(json.dumps(manifest))
    if existing:
        if not args.resume:
            raise ValueError("experiment already exists; use --resume or a new output directory")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("experiment settings changed; use a new output directory")
        if json.loads((output / "inputs/labels.json").read_text()) != labels:
            raise ValueError("labels changed; use a new output directory")
        if not filecmp.cmp(args.rust_binary, output / "inputs/cenalert-detector", shallow=False):
            raise ValueError("detector binary changed; use a new output directory")
        for name in ("run_experiment_matrix.py", "batch_ensemble_tuning.py"):
            if not filecmp.cmp(Path(__file__).with_name(name), output / "inputs" / name,
                               shallow=False):
                raise ValueError("experiment code changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory is nonempty without an experiment manifest")
        (output / "inputs").mkdir(parents=True, exist_ok=True)
        write_json(output / "inputs/labels.json", labels)
        shutil.copy2(args.rust_binary, output / "inputs/cenalert-detector")
        for name in ("run_experiment_matrix.py", "batch_ensemble_tuning.py"):
            shutil.copy2(Path(__file__).with_name(name), output / "inputs" / name)

    details = {}
    for country, source in sorted(by_country.items()):
        destination = output / "inputs" / f"{country}.json"
        candidate = destination.with_suffix(".candidate.json")
        try:
            details[country] = convert_series(source, candidate, country)
            if destination.exists():
                if json.loads(destination.read_text()) != json.loads(candidate.read_text()):
                    raise ValueError(f"series changed for {country}; use a new output directory")
            else:
                if existing:
                    raise ValueError(f"saved input missing for {country}; use a new output directory")
                candidate.replace(destination)
        finally:
            candidate.unlink(missing_ok=True)
    write_json(manifest_path, manifest)
    return sorted(by_country), details


def select_parameters(config: Config, front: Path, parameters: Path, selection: str) -> str:
    solutions = json.loads(front.read_text())
    if solutions and all(objectives[0] == 0 for _, objectives in solutions):
        values, _ = min(solutions, key=lambda solution: (solution[1], solution[0]))
        parameter_values = (ensemble_parameters(values) if config.algorithm == "ensemble" else {
            "window": round(values[0]), "z": values[1], "k": values[2],
            "min_residual_ratio": values[3], "efficiency": values[4],
        })
        write_json(parameters, parameter_values)
        return "no detecting solution; selected deterministic zero-anomaly solution\n"
    return select(config, front, parameters, 1 if selection == "leftmost" else None).stderr


def run_variant(config: Config, country: str, variant: str, details: dict[str, Any],
                selections: list[str], resume: bool) -> dict[str, Any]:
    algorithm, constrained = VARIANTS[variant]
    directory = config.work_directory / "runs" / str(config.seed) / country / variant
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        artifacts = [directory / "pareto-front.json"] + [
            directory / f"{selection}-{kind}.json"
            for selection in selections for kind in ("parameters", "detection")
        ]
        if previous["status"] in ("skipped_all_zero", "no_feasible_solution") or (
            previous["status"] == "completed" and all(path.exists() for path in artifacts)
        ):
            return previous
    config = replace(config, algorithm=algorithm)
    result = dict(country=country, variant=variant, algorithm=algorithm,
                  constrained=constrained, seed=config.seed,
                  effective_seed=country_seed(config.seed, country), **details)
    result["selections"] = []
    try:
        if details["zero_observations"] == details["observations"]:
            result["status"] = "skipped_all_zero"
            return result
        series = config.work_directory / "inputs" / f"{country}.json"
        front = directory / "pareto-front.json"
        start = perf_counter()
        try:
            tuning = tune(config, series, front, country, constrained)
        finally:
            result["tuning_seconds"] = perf_counter() - start
        (directory / "tuning.log").write_text(tuning.stdout + tuning.stderr)
        labels = json.loads(config.labels.read_text())
        required = required_labels(labels, country, details["first_date"],
                                   details["last_date"], config.proximity_days)
        for selection in selections:
            parameters = directory / f"{selection}-parameters.json"
            selection_message = select_parameters(config, front, parameters, selection)
            (directory / f"{selection}-selection.log").write_text(selection_message)
            start = perf_counter()
            detected = detect(config, series, parameters)
            detection_seconds = perf_counter() - start
            write_json(directory / f"{selection}-detection.json", detected)
            anomalies = detected["anomalies"]
            label_results = [
                {"label": label, "matched_starts": [a["start"] for a in anomalies
                 if label_matches(label, date.fromisoformat(a["start"]), config.proximity_days)]}
                for label in required
            ]
            annotated = [{**anomaly, "label_match": matched_label(
                anomaly, labels, country, config.proximity_days)} for anomaly in anomalies]
            result["selections"].append({
                "selection": selection,
                "knee_found": (selection_message.startswith("selected ")
                               if selection == "knee" else None),
                "detection_seconds": detection_seconds,
                "anomaly_count": len(anomalies),
                "country_labels_total": len(required),
                "country_labels_covered": sum(bool(row["matched_starts"]) for row in label_results),
                "unmatched_anomalies": sum(a["label_match"] is None for a in annotated),
                "total_impact": sum(a["impact"] for a in anomalies),
                "labels": label_results, "anomalies": annotated,
            })
        result["status"] = "completed"
    except Exception as error:
        result["status"] = ("no_feasible_solution" if "detects every label" in str(error)
                            else "error")
        result["detail"] = str(error)
    finally:
        write_json(result_path, result)
    return result


def export_results(output: Path, results: list[dict[str, Any]], proximity: int) -> None:
    summaries, anomalies, labels, differences = [], [], [], []
    index = {}
    for run in results:
        base = {key: run[key] for key in ("country", "seed", "effective_seed", "variant",
                                         "algorithm", "constrained", "status")}
        base["tuning_seconds"] = run.get("tuning_seconds")
        if run["status"] != "completed":
            summaries.append({**base, "detail": run.get("detail", "")})
            continue
        for selected in run["selections"]:
            identity = {**base, "selection": selected["selection"]}
            summaries.append({**identity, **{k: v for k, v in selected.items()
                                             if k not in ("labels", "anomalies")}})
            index[(run["seed"], run["country"], selected["selection"], run["variant"])] = selected
            anomalies.extend({**identity, **a, "label_match": json.dumps(a["label_match"])}
                             for a in selected["anomalies"])
            labels.extend({**identity, "date": row["label"]["date"],
                           "description": row["label"].get("description", ""),
                           "matched_starts": ",".join(row["matched_starts"])}
                          for row in selected["labels"])
    for (seed, country, selection, variant), selected in sorted(index.items()):
        if variant not in ("ensemble", "labels_ensemble"):
            continue
        baseline = "original" if variant == "ensemble" else "labels"
        other = index.get((seed, country, selection, baseline))
        if other is None:
            continue
        for direction, source, target in (("ensemble_only", selected, other),
                                           ("chebyshev_only", other, selected)):
            for anomaly in source["anomalies"]:
                nearby = any(abs((date.fromisoformat(anomaly["start"]) -
                                  date.fromisoformat(a["start"])).days) <= proximity
                             for a in target["anomalies"])
                if not nearby:
                    differences.append(dict(seed=seed, country=country, selection=selection,
                                            variant=variant, baseline=baseline, direction=direction,
                                            start=anomaly["start"], end=anomaly["end"],
                                            impact=anomaly["impact"],
                                            label_match=json.dumps(anomaly["label_match"])))
    identity_fields = ["country", "seed", "effective_seed", "variant", "algorithm",
                       "constrained", "status", "selection"]
    write_csv(output / "summary.csv", identity_fields + ["knee_found", "anomaly_count",
              "country_labels_total", "country_labels_covered", "unmatched_anomalies",
              "total_impact", "tuning_seconds", "detection_seconds", "detail"], summaries)
    write_csv(output / "anomalies.csv", identity_fields + ["start", "end", "peak",
              "score", "residual", "impact", "sparse", "label_match"], anomalies)
    write_csv(output / "label-coverage.csv", identity_fields + ["date", "description",
              "matched_starts"], labels)
    write_csv(output / "detector-differences.csv", ["seed", "country", "selection",
              "variant", "baseline", "direction", "start", "end", "impact", "label_match"],
              differences)
    lines = ["# Four-group experiment", "",
             "Raw detections before coalescing, impact filtering, and notification grouping.",
             "Country-label coverage measures known-event preservation. In constrained groups,",
             "these same labels are used during tuning. Unmatched anomalies need review.", "",
             "| Group | Completed | Infeasible | All-zero | Errors |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        counts = [sum(r["variant"] == variant and r["status"] == status for r in results)
                  for status in ("completed", "no_feasible_solution", "skipped_all_zero", "error")]
        lines.append(f"| {variant} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "Counts include every requested seed. Compare matching countries, seeds, and",
              "selection rules in `summary.csv`; failed groups have no comparable detection result.",
              "`detector-differences.csv` lists starts without any counterpart within",
              f"±{proximity} days in the other detector's output, in both directions.",
              "This comparison permits multiple starts to share a counterpart.", "",
              "Seeds control evolutionary operators; the optimizer's initial population is",
              "unseeded. Fresh runs can differ. Saved fronts and parameters preserve this run.",
              "Timing includes subprocess startup and I/O under the configured concurrency.", "",
              "This run tunes and evaluates on the supplied period. It is not a held-out test.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_arguments()
    countries, details = prepare_inputs(args)
    output = args.output_directory.resolve()
    config = Config(labels=output / "inputs/labels.json",
                    rust_binary=output / "inputs/cenalert-detector", work_directory=output,
                    evaluations=args.evaluations, population_size=args.population_size,
                    proximity_days=args.proximity_days, coalesce_sparse_gap_days=1,
                    threads_per_tuning=args.threads_per_tuning, seed=0)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(run_variant, replace(config, seed=seed), country, variant,
                                details[country], args.selections, args.resume)
                for seed in args.seeds for country in countries for variant in VARIANTS]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result['country']} "
                  f"seed={result['seed']} {result['variant']}: {result['status']}", flush=True)
    results.sort(key=lambda r: (r["seed"], r["country"], r["variant"]))
    export_results(output, results, args.proximity_days)
    if any(r["status"] == "error" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
