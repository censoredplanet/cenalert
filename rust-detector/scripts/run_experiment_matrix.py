#!/usr/bin/env python3
"""Run the detector × label-constraint ablation on a fixed dataset."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import statistics
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
    "fixed_3sigma": ("fixed_3sigma", False),
}
SCHEMA_VERSION = 2


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
    parser.add_argument("--train-end", default="2024-12-31", type=date.fromisoformat)
    parser.add_argument("--test-end", default="2026-12-31", type=date.fromisoformat)
    parser.add_argument("--baseline-window", type=int, default=60,
                        help="Previous observations used by the fixed 3-sigma baseline.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate and snapshot inputs without starting tuning.")
    args = parser.parse_args()
    if args.test_end <= args.train_end or args.baseline_window < 2:
        parser.error("test end must follow training end; baseline window must be at least 2")
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
        "train_end": str(args.train_end), "test_end": str(args.test_end),
        "baseline_window": args.baseline_window, "baseline_sigma": 3.0,
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
        rows = json.loads(destination.read_text())
        training = [row for row in rows if row["date"] <= str(args.train_end)]
        evaluation = [row for row in rows if row["date"] <= str(args.test_end)]
        test = [row for row in evaluation if row["date"] > str(args.train_end)]
        write_json(output / "inputs" / f"{country}-train.json", training)
        write_json(output / "inputs" / f"{country}-evaluation.json", evaluation)
        details[country].update(
            train_end=str(args.train_end), test_end=str(args.test_end),
            training_observations=len(training), test_observations=len(test),
            training_maximum=max((row["rate"] for row in training), default=0.0),
            test_first_date=test[0]["date"] if test else None,
            test_last_date=test[-1]["date"] if test else None,
            baseline_window=args.baseline_window,
        )
    write_json(output / "inputs/training-labels.json",
               [label for label in labels if label["date"] <= str(args.train_end)])
    write_json(output / "input-summary.json", details)
    write_json(manifest_path, manifest)
    return sorted(by_country), details


def fixed_three_sigma(rows: list[dict[str, Any]], window: int,
                      reference_maximum: float) -> dict[str, Any]:
    scale = max(1.0, reference_maximum / 100.0)
    anomalies = []
    active = None
    for index in range(window, len(rows)):
        history = [row["rate"] for row in rows[index-window:index]]
        mean = statistics.mean(history)
        std = statistics.pstdev(history)
        value = rows[index]["rate"]
        threshold = mean + 3.0 * std
        if value <= threshold:
            active = None
            continue
        when = rows[index]["date"]
        if active is None:
            active = dict(start=when, end=when, peak=when, peak_value=value / scale,
                          score=(value - mean) / std if std else None,
                          residual=(value - mean) / scale, impact=0.0, sparse=False)
            anomalies.append(active)
        active["end"] = when
        active["impact"] += (value - threshold) / scale
        if value / scale > active["peak_value"]:
            active.update(peak=when, peak_value=value / scale)
    return {"algorithm": "fixed_3sigma", "scale_factor": scale, "anomalies": anomalies}


def evaluate_period(anomalies: list[dict[str, Any]], labels: list[dict[str, Any]],
                    country: str, first: str, last: str, proximity: int) -> dict[str, Any]:
    anomalies = [a for a in anomalies if first <= a["start"] <= last]
    labels = [label for label in labels if first <= label["date"] <= last]
    required = required_labels(labels, country, first, last, 0)
    label_results = [
        {"label": label, "matched_starts": [a["start"] for a in anomalies
         if label_matches(label, date.fromisoformat(a["start"]), proximity)]}
        for label in required
    ]
    annotated = [{**a, "label_match": matched_label(a, labels, country, proximity)}
                 for a in anomalies]
    return dict(anomaly_count=len(anomalies), country_labels_total=len(required),
                country_labels_covered=sum(bool(row["matched_starts"]) for row in label_results),
                unmatched_anomalies=sum(a["label_match"] is None for a in annotated),
                total_impact=sum(a["impact"] for a in anomalies),
                labels=label_results, anomalies=annotated)


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
    if variant == "fixed_3sigma":
        selections = ["fixed"]
    directory = config.work_directory / "runs" / str(config.seed) / country / variant
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        artifacts = ([] if variant == "fixed_3sigma" else [directory / "pareto-front.json"]) + [
            directory / f"{selection}-{kind}.json"
            for selection in selections for kind in ("parameters", "detection")
        ]
        if previous["status"].startswith("skipped_") or previous["status"] == "no_feasible_solution" or (
            previous["status"] == "completed" and all(path.exists() for path in artifacts)
        ):
            return previous
    config = replace(config, algorithm=algorithm)
    result = dict(country=country, variant=variant, algorithm=algorithm,
                  constrained=constrained, seed=config.seed,
                  effective_seed=country_seed(config.seed, country), **details)
    result["selections"] = []
    try:
        if not details["training_observations"] or not details["test_observations"]:
            result["status"] = "skipped_missing_period"
            return result
        if details["zero_observations"] == details["observations"]:
            result["status"] = "skipped_all_zero"
            return result
        if details["training_maximum"] <= 0.0:
            result["status"] = "skipped_zero_training"
            return result
        series = config.work_directory / "inputs" / f"{country}-train.json"
        evaluation_series = config.work_directory / "inputs" / f"{country}-evaluation.json"
        training_labels = config.work_directory / "inputs/training-labels.json"
        front = directory / "pareto-front.json"
        result["tuning_seconds"] = 0.0
        if variant != "fixed_3sigma":
            start = perf_counter()
            try:
                tuning = tune(replace(config, labels=training_labels), series, front, country,
                              constrained and bool(json.loads(training_labels.read_text())))
            finally:
                result["tuning_seconds"] = perf_counter() - start
            (directory / "tuning.log").write_text(tuning.stdout + tuning.stderr)
        labels = json.loads(config.labels.read_text())
        for selection in selections:
            parameters = directory / f"{selection}-parameters.json"
            if selection == "fixed":
                write_json(parameters, {"window": details["baseline_window"], "sigma": 3.0})
                selection_message = "fixed parameters; no tuning\n"
            else:
                selection_message = select_parameters(config, front, parameters, selection)
            (directory / f"{selection}-selection.log").write_text(selection_message)
            start = perf_counter()
            if selection == "fixed":
                detected = fixed_three_sigma(json.loads(evaluation_series.read_text()),
                                             details["baseline_window"], details["training_maximum"])
            else:
                detected = detect(config, evaluation_series, parameters, details["train_end"])
            detection_seconds = perf_counter() - start
            write_json(directory / f"{selection}-detection.json", detected)
            metrics = evaluate_period(detected["anomalies"], labels, country,
                                      details["test_first_date"], details["test_last_date"],
                                      config.proximity_days)
            yearly = []
            for year in range(int(details["test_first_date"][:4]), int(details["test_last_date"][:4])+1):
                first = max(details["test_first_date"], f"{year}-01-01")
                last = min(details["test_last_date"], f"{year}-12-31")
                yearly.append({"year": year, "first_date": first, "last_date": last,
                               **evaluate_period(detected["anomalies"], labels, country,
                                                 first, last, config.proximity_days)})
            result["selections"].append({
                "selection": selection,
                "knee_found": (selection_message.startswith("selected ")
                               if selection == "knee" else None),
                "detection_seconds": detection_seconds,
                **metrics, "yearly": yearly,
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
    summaries, anomalies, labels, differences, annual = [], [], [], [], []
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
                                             if k not in ("labels", "anomalies", "yearly")}})
            annual.extend({**identity, **{k: v for k, v in year.items()
                           if k not in ("labels", "anomalies")}} for year in selected["yearly"])
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
    write_csv(output / "yearly-summary.csv", identity_fields + ["year", "first_date", "last_date",
              "anomaly_count", "country_labels_total", "country_labels_covered",
              "unmatched_anomalies", "total_impact"], annual)
    write_csv(output / "detector-differences.csv", ["seed", "country", "selection",
              "variant", "baseline", "direction", "start", "end", "impact", "label_match"],
              differences)
    lines = ["# Temporal evaluation: four groups and fixed baseline", "",
             "Raw detections before coalescing, impact filtering, and notification grouping.",
             "Coverage uses held-out country labels. Unmatched anomalies need review.",
             "Metrics include intervals starting in the test period. Intervals starting in training",
             "are retained in full detection JSON and excluded from test counts.", "",
             "| Group | Completed | Infeasible | All-zero | Missing period | Zero training | Errors |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        counts = [sum(r["variant"] == variant and r["status"] == status for r in results)
                  for status in ("completed", "no_feasible_solution", "skipped_all_zero",
                                 "skipped_missing_period", "skipped_zero_training", "error")]
        lines.append(f"| {variant} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "Counts include every requested seed. Compare matching countries, seeds, and",
              "selection rules in `summary.csv`; failed groups have no comparable detection result.",
              "`detector-differences.csv` lists starts without any counterpart within",
              f"±{proximity} days in the other detector's output, in both directions.",
              "This comparison permits multiple starts to share a counterpart.", "",
              "Seeds control evolutionary operators; the optimizer's initial population is",
              "unseeded. Fresh runs can differ. Saved fronts and parameters preserve this run.",
              "Timing includes subprocess startup and I/O under the configured concurrency.", "",
              "See `experiment.json` for cutoffs and `input-summary.json` for actual coverage.",
              "`yearly-summary.csv` reports each test year separately; boundary matches use",
              "starts and labels within that year, so annual coverage may differ from pooled coverage.",
              "The fixed baseline uses the preceding observations (including zeros and spikes),",
              "population standard deviation, and a strict value > mean + 3*std decision.",
              "Its score is null when standard deviation is zero; a positive rise still alerts.",
              "Croston and impact scaling use the training maximum. Historical Dashboard stitching",
              "may reflect later data revisions; this is retrospective temporal evaluation.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_arguments()
    countries, details = prepare_inputs(args)
    if args.prepare_only:
        print(f"Prepared {len(countries)} countries in {args.output_directory}")
        return
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
