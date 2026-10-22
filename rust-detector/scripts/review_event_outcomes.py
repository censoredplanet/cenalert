"""Compare event coverage and impact distributions in saved temporal runs."""
import argparse
import json
import subprocess
import tempfile
from bisect import bisect_right
from pathlib import Path

from review_contextualization import NAMES, matches, read, write
from batch_ensemble_tuning import required_labels, ensemble_parameters


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return dict(count=len(values), **{
        name: quantile(values, p) for name, p in
        [("minimum", 0), ("q25", .25), ("median", .5), ("q75", .75), ("maximum", 1)]})


def event_outcomes(labels, raw, retained, reference, proximity):
    ordered = sorted(a["impact"] for a in reference)
    outcomes = []
    for label in labels:
        detected = [a for a in raw if matches(a, label, proximity)]
        kept = [a for a in retained if matches(a, label, proximity)]
        outcomes.append(dict(label=label,
            status="retained" if kept else "filtered" if detected else "missed",
            matches=[dict(start=a["start"], end=a["end"], impact=a["impact"],
                          sparse=a.get("sparse", False),
                          training_percentile=bisect_right(ordered, a["impact"]) / len(ordered)
                          if ordered else None) for a in detected]))
    return outcomes


def analyze(analysis, results):
    training_labels = read(results / "inputs/training-labels.json")
    rows = []
    for row in analysis["countries"]:
        directory = results / f"runs/{analysis['seed']}/{row['country']}/{row['variant']}"
        run = read(directory / "result.json")
        anomalies = read(directory / f"{row['selection']}-detection.json")["anomalies"]
        training = [a for a in anomalies if a["start"] <= analysis["train_end"]]
        reference = [a for a in training if a["end"] <= analysis["train_end"]]
        train_required = required_labels(training_labels, row["country"], "0001-01-01",
                                         analysis["train_end"], 0)
        test_required = required_labels(analysis["evaluation_labels"], row["country"],
                                        run["test_first_date"], run["test_last_date"], 0)
        train = event_outcomes(train_required, training, training, reference,
                               analysis["country_proximity_days"])
        test = event_outcomes(test_required, row["raw_anomalies"], row["retained_anomalies"],
                              reference, analysis["country_proximity_days"])
        assert sum(e["status"] != "missed" for e in train) == row["training_labels_covered"]
        assert sum(e["status"] != "missed" for e in test) == row["raw_covered"]
        assert sum(e["status"] == "retained" for e in test) == row["retained_covered"]
        rows.append(dict(country=row["country"], variant=row["variant"], selection=row["selection"],
                         training=train, test=test, threshold=row["threshold"]))
    summaries = []
    for variant, selection in sorted({(r["variant"], r["selection"]) for r in rows}):
        selected = [r for r in analysis["countries"] if r["variant"] == variant
                    and r["selection"] == selection and r["country"] in analysis["comparison_countries"]]
        for stage in ["raw_anomalies", "retained_anomalies"]:
            for subset in ["all", "country_label_matched"]:
                values = []
                for r in selected:
                    labels = next(e["test"] for e in rows if e["country"] == r["country"]
                                  and e["variant"] == variant and e["selection"] == selection)
                    values.extend(a["impact"] for a in r[stage] if subset == "all" or any(
                        matches(a, e["label"], analysis["country_proximity_days"]) for e in labels))
                summaries.append(dict(variant=variant, selection=selection, stage=stage, subset=subset,
                                      **distribution(values)))
    return dict(source=str(results), comparison_countries=analysis["comparison_countries"],
                excluded=analysis["excluded"],
                quantile_method="linear interpolation at (n-1)*p",
                percentile_reference="same-country, same-configuration raw training anomalies ending by cutoff",
                countries=rows, impact_distributions=summaries)


def report(result):
    lines = ["# Event-level evaluation", "",
             "Impact summaries use the common completed countries. Country-label coverage uses the "
             "saved matching windows. Each event record includes the matched anomaly and its impact "
             "percentile among that country's raw training anomalies under the same configuration.", "",
             "## Test impact distributions", "",
             "| Configuration | Stage | Subset | Count | Q25 | Median | Q75 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["impact_distributions"]:
        if row["selection"] not in ("knee", "fixed"):
            continue
        fmt = lambda value: "—" if value is None else f"{value:.3g}"
        lines.append(f"| {NAMES[row['variant']]} | {row['stage']} | {row['subset']} | {row['count']} | "
                     + " | ".join(fmt(row[k]) for k in ("q25", "median", "q75")) + " |")
    lines += ["", "## Russia: missed country labels", ""]
    for row in result["excluded"]:
        if row["country"] == "RU":
            lines += [f"{NAMES[row['variant']]}: {row['status']}. The saved failed run contains no "
                      "candidate parameters or per-label diagnostics.", ""]
    for row in result["countries"]:
        if row["country"] != "RU" or row["selection"] not in ("knee", "fixed"):
            continue
        lines += [f"### {NAMES[row['variant']]}", ""]
        for period in ("training", "test"):
            missing = [e["label"] for e in row[period] if e["status"] == "missed"]
            lines.append(f"{period}: {len(missing)} missed out of {len(row[period])}.")
            lines += [f"- {l['date']}: {l.get('title') or l.get('description', '')}" for l in missing]
            lines.append("")
    lines += ["## Labeled ensemble: test events removed by impact filtering", ""]
    for row in result["countries"]:
        if row["variant"] != "labels_ensemble" or row["selection"] != "knee":
            continue
        for event in row["test"]:
            if event["status"] == "filtered":
                label = event["label"]
                details = "; ".join(f"{a['start']}: IF {a['impact']:.4g}, training percentile "
                                    + (f"{a['training_percentile']:.1%}" if a['training_percentile'] is not None else "unavailable")
                                    for a in event["matches"])
                lines.append(f"- {row['country']} {label['date']}: {label.get('title') or label.get('description', '')}. "
                             f"{details}; filtering threshold {row['threshold']:.4g}.")
    for diagnostic in result.get("pareto_diagnostics", []):
        lines += ["", f"## {diagnostic['country']}: saved unconstrained ensemble front", "",
                  f"Evaluated {diagnostic['solutions']} distinct parameter sets. Best simultaneous training-label "
                  f"coverage: {diagnostic['best_coverage']}/{diagnostic['required_labels']}.", "",
                  "Training labels never matched by any saved candidate:"]
        lines += [f"- {l['date']}: {l.get('title') or l.get('description', '')}"
                  for l in diagnostic["never_detected"]]
        if not diagnostic["never_detected"]:
            lines.append("Every required label is matched by at least one saved candidate.")
        lines += ["", "Missed-label combinations among the best simultaneous-coverage candidates:"]
        combinations = sorted({tuple((l["date"], l.get("title") or l.get("description", ""))
                                     for l in c["missed_labels"]) for c in diagnostic["best_candidates"]})
        lines += ["- " + "; ".join(f"{day}: {title}" for day, title in group) for group in combinations]
    return "\n".join(lines) + "\n"


def review_front(country, results, binary):
    config = read(results / "experiment.json")
    seed, = config["seeds"]
    front = read(results / f"runs/{seed}/{country}/ensemble/pareto-front.json")
    required = required_labels(read(results / "inputs/training-labels.json"), country,
                               "0001-01-01", config["train_end"], 0)
    seen, candidates = set(), []
    with tempfile.TemporaryDirectory() as directory:
        parameters = Path(directory) / "parameters.json"
        for values, _ in front:
            params = ensemble_parameters(values)
            key = json.dumps(params, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            write(parameters, params)
            output = subprocess.run([str(binary), "detect", "--algorithm", "ensemble",
                "--input", str(results / f"inputs/{country}.json"), "--parameters", str(parameters),
                "--reference-end", config["train_end"]], check=True, capture_output=True, text=True)
            anomalies = [a for a in json.loads(output.stdout)["anomalies"] if a["start"] <= config["train_end"]]
            covered = [i for i, l in enumerate(required)
                       if any(matches(a, l, config["proximity_days"]) for a in anomalies)]
            candidates.append(dict(parameters=params, covered=covered, anomalies=len(anomalies)))
    union = {i for c in candidates for i in c["covered"]}
    best = max((len(c["covered"]) for c in candidates), default=0)
    return dict(country=country, solutions=len(candidates), required_labels=len(required), best_coverage=best,
                never_detected=[l for i, l in enumerate(required) if i not in union],
                best_candidates=[dict(parameters=c["parameters"], anomalies=c["anomalies"],
                    missed_labels=[l for i, l in enumerate(required) if i not in c["covered"]])
                    for c in candidates if len(c["covered"]) == best])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextualization", type=Path, required=True, help="saved analysis.json")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-front", nargs="*", default=[], metavar="CC",
                        help="evaluate saved unconstrained ensemble Pareto candidates for these countries")
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "target/release/cenalert-detector")
    args = parser.parse_args()
    result = analyze(read(args.contextualization), args.results.resolve())
    result["pareto_diagnostics"] = [review_front(c, args.results.resolve(), args.binary.resolve())
                                    for c in args.review_front]
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "event-outcomes.json", result)
    (args.output / "event-review.md").write_text(report(result))


if __name__ == "__main__":
    main()
