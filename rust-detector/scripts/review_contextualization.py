"""Review contextualization of saved ensemble outputs without retuning."""
import argparse
import csv
import json
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

from batch_ensemble_tuning import label_applies, label_matches, required_labels


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def starts(anomaly):
    return anomaly.get("component_starts") or [anomaly["start"]]


def matches(anomaly, label, proximity):
    return any(label_matches(label, date.fromisoformat(day), proximity)
               for day in starts(anomaly))


def coverage(anomalies, labels, proximity):
    return [label for label in labels if any(matches(a, label, proximity) for a in anomalies)]


def retain_by_impact(anomalies, threshold):
    if threshold is None:
        return list(anomalies)
    return [a for a in anomalies if a["impact"] >= threshold]


def evaluation_labels(snapshot, replacement, cutoff):
    def training(items):
        return sorted(json.dumps(l, sort_keys=True) for l in items if l["date"] <= cutoff)
    if training(snapshot) != training(replacement):
        raise ValueError("evaluation label updates must preserve training labels")
    return replacement


def notification_matches(grouping, labels_by_country, proximity):
    def annotate(anomaly):
        labels = labels_by_country[anomaly["country"]]
        return dict(country=anomaly["country"], start=anomaly["start"],
                    labels=[l for l in labels if l["scope"] in ("country", "global")
                            and label_applies(l, anomaly["country"])
                            and matches(anomaly, l, proximity)])
    individuals = [annotate(a) for a in grouping["individual_notifications"]]
    groups = [dict(date=g["date"], members=[annotate(a) for a in g["anomalies"]])
              for g in grouping["grouped_notifications"]]
    individual_matched = sum(bool(a["labels"]) for a in individuals)
    group_any = sum(any(a["labels"] for a in g["members"]) for g in groups)
    group_all = sum(all(a["labels"] for a in g["members"]) for g in groups)
    return dict(total=len(individuals) + len(groups),
                matched=individual_matched + group_any,
                individual_matched=individual_matched, groups_any_matched=group_any,
                groups_all_members_matched=group_all,
                individuals=individuals, groups=groups)


def unmatched_notification_rows(grouping):
    """Keep each constituent interval, sharing an ID for a grouped notification."""
    matching = grouping["label_matching"]
    rows = []

    def append(identifier, kind, when, countries, anomalies):
        for a in anomalies:
            rows.append(dict(notification_id=identifier, notification_type=kind,
                notification_date=when, countries=";".join(countries),
                country=a["country"], start=a["start"], end=a["end"], peak=a["peak"],
                peak_value=a.get("peak_value"), impact=a["impact"],
                sparse=a.get("sparse", False), component_starts=";".join(starts(a))))

    for anomaly, annotated in zip(grouping["individual_notifications"], matching["individuals"], strict=True):
        if not annotated["labels"]:
            append(f"{anomaly['country']}-{anomaly['start']}", "individual", anomaly["start"],
                   [anomaly["country"]], [anomaly])
    for group, annotated in zip(grouping["grouped_notifications"], matching["groups"], strict=True):
        if not any(member["labels"] for member in annotated["members"]):
            append(f"group-{group['date']}", "cross_country", group["date"],
                   group["countries"], group["anomalies"])
    return sorted(rows, key=lambda r: (r["notification_date"], r["notification_id"], r["country"]))


def write_unmatched_notifications(path, grouping):
    fields = ["notification_id", "notification_type", "notification_date", "countries",
              "country", "start", "end", "peak", "peak_value", "impact", "sparse", "component_starts"]
    rows = unmatched_notification_rows(grouping)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert len({r["notification_id"] for r in rows}) == (
        grouping["label_matching"]["total"] - grouping["label_matching"]["matched"])


def filter_output(binary, country, anomalies, labels_path, proximity, gap, directory):
    source, output = directory / "input.json", directory / "output.json"
    write(source, {"countries": [country], "anomalies": anomalies})
    subprocess.run([str(binary), "filter", "--input", str(source), "--labels", str(labels_path),
                    "--output", str(output), "--proximity-days", str(proximity),
                    "--coalesce-sparse-gap-days", str(gap)], check=True, capture_output=True, text=True)
    result = read(output)
    # Recover the complete coalesced set, before the CLI's impact filtering.
    coalesced = sorted(result["anomalies"] + [r["anomaly"] for r in result["discarded"]],
                       key=lambda a: a["start"])
    return coalesced, result["stages"][0].get("impact_threshold")


def ordering_cases(anomalies, labels, country, proximity):
    cases = []
    for anomaly in anomalies:
        candidates = [(min(abs((date.fromisoformat(s) - date.fromisoformat(l["date"])).days)
                           for s in starts(anomaly)), l)
                      for l in labels if label_applies(l, country) and matches(anomaly, l, proximity)]
        local = [x for x in candidates if x[1]["scope"] == "country"]
        global_ = [x for x in candidates if x[1]["scope"] == "global"]
        if local and global_ and min(x[0] for x in global_) < min(x[0] for x in local):
            cases.append(dict(country=country, start=anomaly["start"],
                              country_labels=[dict(distance=d, label=l) for d, l in local],
                              global_labels=[dict(distance=d, label=l) for d, l in global_]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-gap-days", type=int, default=1)
    parser.add_argument("--evaluation-labels", type=Path,
                        help="updated labels for test-period attribution; training labels must be unchanged")
    args = parser.parse_args()
    if args.sparse_gap_days < 0:
        parser.error("sparse gap must be non-negative")
    root = args.results.resolve()
    config = read(root / "experiment.json")
    if len(config["seeds"]) != 1:
        parser.error("use a single-seed experiment")
    seed = config["seeds"][0]
    cutoff, end = config["train_end"], config["test_end"]
    first = str(date.fromisoformat(cutoff) + timedelta(days=1))
    proximity = config["proximity_days"]
    labels = read(root / "inputs/labels.json")
    if args.evaluation_labels:
        labels = evaluation_labels(labels, read(args.evaluation_labels), cutoff)
    training_labels = root / "inputs/training-labels.json"
    if any(label["date"] > cutoff for label in read(training_labels)):
        raise ValueError("training-label snapshot contains post-cutoff labels")
    test_labels = [l for l in labels if first <= l["date"] <= end]
    binary = root / "inputs/cenalert-detector"
    args.output.mkdir(parents=True, exist_ok=True)
    countries, excluded, ordering = [], [], []
    labels_by_country = {}
    candidates = {s: [] for s in ["knee", "leftmost"]}
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for country in config["countries"]:
            directory = root / f"runs/{seed}/{country}/labels_ensemble"
            result = read(directory / "result.json")
            if result["status"] != "completed":
                excluded.append(dict(country=country, status=result["status"]))
                continue
            period_labels = [l for l in test_labels
                             if result["test_first_date"] <= l["date"] <= result["test_last_date"]]
            labels_by_country[country] = period_labels
            required = required_labels(period_labels, country, result["test_first_date"],
                                       result["test_last_date"], 0)
            for selection in candidates:
                detection = read(directory / f"{selection}-detection.json")
                all_anomalies = detection["anomalies"]
                training = [a for a in all_anomalies if a["end"] <= cutoff]
                raw = [a for a in all_anomalies if first <= a["start"] <= end]
                _, threshold = filter_output(binary, country, training, training_labels,
                                             proximity, args.sparse_gap_days, scratch)
                coalesced, _ = filter_output(binary, country, raw, training_labels,
                                             proximity, args.sparse_gap_days, scratch)
                retained = retain_by_impact(coalesced, threshold)
                raw_covered = coverage(raw, required, proximity)
                recorded = next(s for s in result["selections"] if s["selection"] == selection)
                assert len(raw) == recorded["anomaly_count"]
                if not args.evaluation_labels:
                    assert (len(required), len(raw_covered)) == (
                        recorded["country_labels_total"], recorded["country_labels_covered"])
                retained_covered = coverage(retained, required, proximity)
                covered_coalesced = coverage(coalesced, required, proximity)
                assert covered_coalesced == raw_covered
                countries.append(dict(country=country, selection=selection,
                    threshold=threshold, training_anomalies=len(training),
                    raw=len(raw), raw_sparse=sum(a.get("sparse", False) for a in raw),
                    coalesced=len(coalesced), retained=len(retained),
                    labels_total=len(required), raw_covered=len(raw_covered),
                    retained_covered=len(retained_covered),
                    lost_labels=[l for l in raw_covered if l not in retained_covered],
                    excluded_boundary_intervals=[a for a in all_anomalies if a["start"] <= cutoff < a["end"]],
                    parameters=detection["parameters"], raw_anomalies=raw,
                    coalesced_anomalies=coalesced, retained_anomalies=retained))
                candidates[selection].extend(dict(country=country, **a) for a in retained)
                if selection == "knee":
                    ordering.extend(ordering_cases(raw, period_labels, country, proximity))
        grouping = {}
        for selection, anomalies in candidates.items():
            source = scratch / "candidates.json"
            write(source, anomalies)
            grouping[selection] = {}
            for minimum in [4, 5]:
                destination = scratch / "grouped.json"
                subprocess.run([str(binary), "group-cross-country", "--input", str(source),
                                "--output", str(destination), "--minimum-countries", str(minimum)],
                               check=True, capture_output=True, text=True)
                grouping[selection][str(minimum)] = read(destination)
    summary = {}
    for selection in candidates:
        rows = [r for r in countries if r["selection"] == selection]
        summary[selection] = {key: sum(r[key] for r in rows) for key in
                              ["raw", "raw_sparse", "coalesced", "retained", "labels_total",
                               "raw_covered", "retained_covered"]}
        summary[selection]["countries"] = len(rows)
        summary[selection]["without_calibration"] = [r["country"] for r in rows if r["threshold"] is None]
        summary[selection]["notifications"] = {
            minimum: g["individual_notification_count"] + g["grouped_notification_count"]
            for minimum, g in grouping[selection].items()}
        summary[selection]["notification_matching"] = {}
        for minimum, g in grouping[selection].items():
            detail = notification_matches(g, labels_by_country, proximity)
            g["label_matching"] = detail
            summary[selection]["notification_matching"][minimum] = {
                k: v for k, v in detail.items() if k not in ("individuals", "groups")}
    analysis = dict(source=str(root), train_end=cutoff, test_end=end, seed=seed,
                    sparse_gap_days=args.sparse_gap_days, threshold_reference="training",
                    summary=summary, excluded=excluded, countries=countries,
                    ordering_cases=ordering, grouping=grouping)
    analysis["evaluation_labels"] = labels
    write(args.output / "analysis.json", analysis)
    write_unmatched_notifications(args.output / "unmatched-notifications.csv", grouping["knee"]["5"])
    lines = ["# Contextualization of held-out ensemble detections", "",
             "All completed label-constrained ensemble countries are included. Impact thresholds use only "
             "label-matched, coalesced training anomalies. Training intervals crossing the cutoff are excluded "
             "from calibration. Test labels are used for evaluation and attribution. A country with no "
             "matched training anomaly retains all coalesced candidates. "
             f"Sparse gaps allow {args.sparse_gap_days} intervening normal day(s). "
             "Grouping uses exact anomaly start dates and preserves constituent anomalies.", "",
             "| Selection | Countries | Raw | Coalesced | Impact retained | Alerts (4 countries) | Alerts (5 countries) | Country-label coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for selection, s in summary.items():
        lines.append(f"| {selection} | {s['countries']} | {s['raw']} | {s['coalesced']} | {s['retained']} | "
                     f"{s['notifications']['4']} | {s['notifications']['5']} | "
                     f"{s['raw_covered']}/{s['labels_total']} to {s['retained_covered']}/{s['labels_total']} |")
    lines += ["", "## Notification label matching", "",
              "A grouped notification counts as matched when at least one constituent matches a country "
              "or global label. Full constituent coverage is reported separately.", ""]
    for selection, s in summary.items():
        for minimum, m in s["notification_matching"].items():
            lines.append(f"- {selection}, grouping at {minimum} countries: {m['matched']}/{m['total']} "
                         f"notifications matched ({m['individual_matched']} individual, "
                         f"{m['groups_any_matched']} grouped; "
                         f"{m['groups_all_members_matched']} groups have every constituent matched).")
    lines += ["", "## Labels removed by impact filtering", ""]
    for r in countries:
        for l in r["lost_labels"]:
            lines.append(f"- {r['selection']}: {r['country']} {l['date']} — {l.get('description', '')}")
    lines += ["", "## Suriname", ""]
    for r in countries:
        if r["country"] == "SR":
            lines.append(f"- {r['selection']}: {r['raw']} raw ({r['raw_sparse']} sparse), "
                         f"{r['coalesced']} coalesced, {r['retained']} impact-retained. "
                         f"Training-calibrated threshold: {r['threshold']}. "
                         f"Training anomaly count: {r['training_anomalies']}. "
                         f"Residual ratio: {r['parameters']['min_residual_ratio']}. "
                         f"Maximum coalesced test impact: {max((a['impact'] for a in r['coalesced_anomalies']), default=0)}.")
    lines += ["", "## Country/global attribution conflicts", ""]
    for case in ordering:
        lines.append(f"- {case['country']} {case['start']}: " + "; ".join(
            f"{kind}: {item['label']['date']} {item['label'].get('description', '')} ({item['distance']} days)"
            for kind in ["country_labels", "global_labels"] for item in case[kind]))
    lines += ["", "unmatched-notifications.csv lists unmatched knee/fallback notifications with grouping "
              "at five countries. Each row is a constituent anomaly; rows sharing a notification_id "
              "belong to one cross-country notification.", "",
              "Per-country thresholds, retained/discarded-by-comparison anomalies, lost labels, and "
              "all notification groups are recorded in analysis.json."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
