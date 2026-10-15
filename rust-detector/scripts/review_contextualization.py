"""Review contextualization of saved temporal detections without retuning."""
import argparse
import csv
import json
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

from batch_ensemble_tuning import label_applies, label_matches, required_labels

VARIANTS = ["original", "labels", "ensemble", "labels_ensemble", "fixed_3sigma"]
NAMES = {"original": "Single detector", "labels": "Single detector + labels",
         "ensemble": "Ensemble", "labels_ensemble": "Ensemble + labels", "fixed_3sigma": "Fixed 3σ"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def starts(anomaly):
    return anomaly.get("component_starts") or [anomaly["start"]]


def transport_scores(anomalies):
    """The CLI requires a numeric score; contextualization never filters on it."""
    return [dict(a, score=0.0) if a.get("score") is None else a for a in anomalies]


def restore_null_scores(anomalies, originals):
    missing = {(a.get("country"), s) for a in originals if a.get("score") is None for s in starts(a)}
    for a in anomalies:
        if any((a.get("country"), s) in missing for s in starts(a)):
            a["score"] = None


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


def common_completed_countries(runs, countries, variants):
    return [c for c in countries if all(runs[c, v]["status"] == "completed" for v in variants)]


def calibration_evidence(anomalies, labels, country, proximity, threshold):
    applicable = [l for l in labels if label_applies(l, country)]
    matched = [dict(anomaly=a, labels=[l for l in applicable if matches(a, l, proximity)])
               for a in anomalies]
    matched = [a for a in matched if a["labels"]]
    expected = min((a["anomaly"]["impact"] for a in matched), default=None)
    if expected != threshold:
        raise ValueError(f"calibration disagrees with Rust filter for {country}: {expected} != {threshold}")
    return dict(applicable_training_labels=len(applicable), matched_training_anomalies=len(matched),
                threshold_sources=[a for a in matched if a["anomaly"]["impact"] == threshold])


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
    write(source, {"countries": [country], "anomalies": transport_scores(anomalies)})
    command = [str(binary), "filter", "--input", str(source), "--labels", str(labels_path),
               "--output", str(output), "--proximity-days", str(proximity),
               "--coalesce-sparse-gap-days", str(gap)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"filter failed for {country} ({len(anomalies)} anomalies): {result.stderr.strip()}")
    result = read(output)
    # Recover the complete coalesced set, before the CLI's impact filtering.
    coalesced = sorted(result["anomalies"] + [r["anomaly"] for r in result["discarded"]],
                       key=lambda a: a["start"])
    restore_null_scores(coalesced, anomalies)
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


def aggregate_rows(rows, binary, labels_by_country, proximity, scratch):
    candidates = [dict(country=r["country"], **a) for r in rows for a in r["retained_anomalies"]]
    source = scratch / "candidates.json"
    write(source, transport_scores(candidates))
    grouping = {}
    for minimum in [4, 5]:
        destination = scratch / "grouped.json"
        subprocess.run([str(binary), "group-cross-country", "--input", str(source),
                        "--output", str(destination), "--minimum-countries", str(minimum)],
                       check=True, capture_output=True, text=True)
        grouping[str(minimum)] = read(destination)
        restore_null_scores(grouping[str(minimum)]["individual_notifications"], candidates)
        for group in grouping[str(minimum)]["grouped_notifications"]:
            restore_null_scores(group["anomalies"], candidates)
        grouping[str(minimum)]["label_matching"] = notification_matches(
            grouping[str(minimum)], labels_by_country, proximity)
    summary = {key: sum(r[key] for r in rows) for key in
               ["raw", "raw_sparse", "coalesced", "retained", "training_labels_total", "training_labels_covered", "labels_total",
                "raw_covered", "retained_covered"]}
    summary.update(countries=len(rows), variant=rows[0]["variant"] if rows else None,
                   selection=rows[0]["selection"] if rows else None,
                   low_impact_retained=sum(a["impact"] < 1 for a in candidates),
                   without_calibration=[r["country"] for r in rows if r["threshold"] is None])
    summary["notifications"] = {
        minimum: g["individual_notification_count"] + g["grouped_notification_count"]
        for minimum, g in grouping.items()}
    summary["notification_matching"] = {
        minimum: {k: v for k, v in g["label_matching"].items() if k not in ("individuals", "groups")}
        for minimum, g in grouping.items()}
    return summary, grouping


def low_impact_rows(countries, common):
    rows = []
    for record in countries:
        for anomaly in record["retained_anomalies"]:
            if anomaly["impact"] >= 1:
                continue
            rows.append(dict(
                variant=record["variant"], selection=record["selection"], country=record["country"],
                in_common_countries=record["country"] in common, anomaly=anomaly,
                threshold=record["threshold"],
                reason="no_matched_training_anomaly" if record["threshold"] is None else "above_calibrated_threshold",
                matched_training_anomalies=record["calibration"]["matched_training_anomalies"],
                threshold_sources=record["calibration"]["threshold_sources"]))
    return rows


def write_low_impact_review(path, analysis):
    key = "labels_ensemble/knee" if analysis["all_configurations"] else "knee"
    matching = analysis["available_grouping"][key]["5"]["label_matching"]
    annotations = matching["individuals"] + [a for g in matching["groups"] for a in g["members"]]
    labels = {(a["country"], a["start"]): a["labels"] for a in annotations}
    fields = ["country", "start", "end", "impact", "sparse", "impact_threshold", "reason",
              "training_anomaly_starts", "calibration_label_dates", "calibration_events",
              "calibration_scopes", "test_label_matches"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in analysis["low_impact_retained"]:
            if r["variant"] != "labels_ensemble" or r["selection"] != "knee":
                continue
            a = r["anomaly"]
            sources = r["threshold_sources"]
            events = [l for source in sources for l in source["labels"]]
            writer.writerow(dict(country=r["country"], start=a["start"], end=a["end"],
                impact=a["impact"], sparse=a.get("sparse", False), impact_threshold=r["threshold"],
                reason=r["reason"], training_anomaly_starts="; ".join(s["anomaly"]["start"] for s in sources),
                calibration_label_dates="; ".join(l["date"] for l in events),
                calibration_events="; ".join(l.get("title") or l.get("description", "") for l in events),
                calibration_scopes="; ".join(l["scope"] for l in events),
                test_label_matches="; ".join(f"{l['date']} {l.get('description', '')}"
                                             for l in labels[r["country"], a["start"]])))


def selection_order(summary):
    return sorted(summary.values(), key=lambda s: (
        ["knee", "leftmost", "fixed"].index(s["selection"]), VARIANTS.index(s["variant"])))


def infeasible_note(excluded):
    countries = sorted({r["country"] for r in excluded if r["status"] == "no_feasible_solution"})
    return (" *Marked configurations found no feasible solution for " + ", ".join(countries) +
            " within the evaluation budget.") if countries else ""


def write_paper_table(path, summary, excluded=()):
    failed = {r["variant"] for r in excluded if r["status"] == "no_feasible_solution"}
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Post-contextualization results. Matched counts alerts with at least one country or global label match; Test labels reports retained country-label coverage. K and L follow Table~\ref{tab:temporal-results}." + infeasible_note(excluded) + "}",
             r"\label{tab:contextualization}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Configuration & Selection & Countries & Raw & Alerts & Matched & Test labels \\", r"\midrule"]
    previous_selection = None
    for s in selection_order(summary):
        if previous_selection is not None and s["selection"] != previous_selection:
            lines.append(r"\midrule")
        previous_selection = s["selection"]
        m = s["notification_matching"]["5"]
        name = NAMES[s["variant"]].replace("Fixed 3σ", r"Fixed $3\sigma$")
        if s["variant"] in failed:
            name += r"$^{*}$"
        rule = {"knee": "K", "leftmost": "L", "fixed": "--"}[s["selection"]]
        lines.append(f"{name} & {rule} & {s['countries']} & {s['raw']} & {m['total']} & {m['matched']} & "
                     f"{s['retained_covered']}/{s['labels_total']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def write_detection_table(path, summary, excluded):
    failed = {r["variant"] for r in excluded if r["status"] == "no_feasible_solution"}
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Raw detections and country-label coverage. K: knee/fallback; L: leftmost." + infeasible_note(excluded) + "}",
             r"\label{tab:temporal-results}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Configuration & Selection & Countries & Train labels & Anomalies & Test labels \\", r"\midrule"]
    previous_selection = None
    for s in selection_order(summary):
        if previous_selection is not None and s["selection"] != previous_selection:
            lines.append(r"\midrule")
        previous_selection = s["selection"]
        name = NAMES[s["variant"]].replace("Fixed 3σ", r"Fixed $3\sigma$")
        if s["variant"] in failed:
            name += r"$^{*}$"
        rule = {"knee": "K", "leftmost": "L", "fixed": "--"}[s["selection"]]
        lines.append(f"{name} & {rule} & {s['countries']} & {s['training_labels_covered']}/{s['training_labels_total']} & "
                     f"{s['raw']} & {s['raw_covered']}/{s['labels_total']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def calibration_review_rows(countries):
    rows = []
    for r in countries:
        count = r["calibration"]["matched_training_anomalies"]
        if count > 1:
            continue
        sparse = lambda anomalies: sum(a.get("sparse", False) for a in anomalies)
        sources = r["calibration"]["threshold_sources"]
        rows.append(dict(configuration=r["variant"], selection=r["selection"], country=r["country"],
            explained_training_anomalies=count, training_anomalies=r["training_anomalies"],
            coalesced_training_anomalies=r["coalesced_training_anomalies"], impact_threshold=r["threshold"],
            test_raw=r["raw"], test_coalesced=r["coalesced"], test_retained=r["retained"],
            impact_filtered=r["coalesced"]-r["retained"],
            sparse_coalesced=sparse(r["coalesced_anomalies"]),
            sparse_filtered=sparse(r["coalesced_anomalies"])-sparse(r["retained_anomalies"]),
            country_labels_before=r["raw_covered"], country_labels_after=r["retained_covered"],
            calibration_events="; ".join(f"{l['date']} {l.get('description', '')}" for s in sources for l in s["labels"])))
    return rows


def write_calibration_review(path, countries):
    rows = calibration_review_rows(countries)
    fields = ["configuration", "selection", "country", "explained_training_anomalies",
              "training_anomalies", "coalesced_training_anomalies", "impact_threshold",
              "test_raw", "test_coalesced", "test_retained", "impact_filtered",
              "sparse_coalesced", "sparse_filtered", "country_labels_before",
              "country_labels_after", "calibration_events"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_configuration_summary(path, summary):
    fields = ["configuration", "selection", "countries", "raw_anomalies", "coalesced_anomalies",
              "impact_retained", "notifications", "matched_notifications", "unmatched_notifications",
              "training_labels_covered", "training_labels_total",
              "country_labels_before", "country_labels_after", "country_labels_total", "retained_impact_below_one"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for s in summary.values():
            m = s["notification_matching"]["5"]
            writer.writerow(dict(configuration=s["variant"], selection=s["selection"], countries=s["countries"],
                raw_anomalies=s["raw"], coalesced_anomalies=s["coalesced"], impact_retained=s["retained"],
                notifications=m["total"], matched_notifications=m["matched"],
                unmatched_notifications=m["total"] - m["matched"], country_labels_before=s["raw_covered"],
                training_labels_covered=s["training_labels_covered"], training_labels_total=s["training_labels_total"],
                country_labels_after=s["retained_covered"], country_labels_total=s["labels_total"],
                retained_impact_below_one=s["low_impact_retained"]))


def report_lines(analysis):
    mode = "Common completed countries" if analysis["all_configurations"] else "All completed ensemble countries"
    lines = ["# Contextualization of held-out detections", "",
             f"{mode}. Impact thresholds use coalesced training anomalies matched to training labels. "
             "Training intervals crossing the cutoff are excluded from calibration. "
             "Thresholds remain fixed during testing; countries without a calibration match retain all candidates. "
             f"Sparse coalescing allows {analysis['sparse_gap_days']} intervening normal day(s). "
             "Cross-country grouping uses exact start dates and preserves each constituent anomaly.", "",
             "| Configuration | Rule | Countries | Raw | Coalesced | Impact retained | Alerts (≥5 countries) | Matched alerts | Country labels before → after | IF < 1 retained |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for s in analysis["summary"].values():
        m = s["notification_matching"]["5"]
        lines.append(f"| {NAMES[s['variant']]} | {s['selection']} | {s['countries']} | {s['raw']} | "
                     f"{s['coalesced']} | {s['retained']} | {m['total']} | {m['matched']} | "
                     f"{s['raw_covered']}/{s['labels_total']} → {s['retained_covered']}/{s['labels_total']} | "
                     f"{s['low_impact_retained']} |")
    lines += ["", "A notification is matched if any constituent matches a country or global label. "
              "The JSON also reports groups with every constituent matched and grouping at four countries.", "",
              "Country matching uses the configured window; global matching is capped at ±1 day.", "",
              "## Countries with zero or one explained training anomaly", "",
              "Counts refer to coalesced training anomalies matching country or global labels. "
              "Filtered counts exclude reductions due to coalescing and cross-country grouping.", "",
              "| Configuration | Selection | Explained training anomalies | Countries | Coalesced test anomalies | Impact filtered | Sparse filtered |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    calibration = calibration_review_rows(analysis["countries"])
    for s in analysis["available_summary"].values():
        for count in [0, 1]:
            rows = [r for r in calibration if r["configuration"] == s["variant"]
                    and r["selection"] == s["selection"] and r["explained_training_anomalies"] == count]
            lines.append(f"| {NAMES[s['variant']]} | {s['selection']} | {count} | {len(rows)} | "
                         f"{sum(r['test_coalesced'] for r in rows)} | {sum(r['impact_filtered'] for r in rows)} | "
                         f"{sum(r['sparse_filtered'] for r in rows)} |")
    lines += ["",
              "## Low-impact retained anomalies", "",
              "The following countries have retained IF < 1 anomalies under label-constrained ensemble "
              "knee/fallback selection, including all available countries.", "",
              "| Country | Retained IF < 1 | Threshold | Training anomaly and matched event(s) setting the threshold |",
              "| --- | ---: | ---: | --- |"]
    for r in analysis["countries"]:
        if r["variant"] != "labels_ensemble" or r["selection"] != "knee":
            continue
        count = sum(a["impact"] < 1 for a in r["retained_anomalies"])
        if not count:
            continue
        threshold = "none" if r["threshold"] is None else f"{r['threshold']:.9g}"
        evidence = "; ".join(
            f"{item['anomaly']['start']} (IF {item['anomaly']['impact']:.9g}): " +
            ", ".join(f"{l['date']} [{l['scope']}] {l.get('description', '')}"
                      for l in item["labels"])
            for item in r["calibration"]["threshold_sources"])
        lines.append(f"| {r['country']} | {count} | {threshold} | {evidence or 'No training anomaly matched a label'} |")
    lines += ["", "## Country labels removed by impact filtering", ""]
    for r in analysis["countries"]:
        if r["country"] not in analysis["comparison_countries"]:
            continue
        for l in r["lost_labels"]:
            lines.append(f"- {r['variant']}/{r['selection']}: {r['country']} {l['date']} — {l.get('description', '')}")
    lines += ["", "## Available-country ensemble result", ""]
    key = "labels_ensemble/knee" if analysis["all_configurations"] else "knee"
    s = analysis["available_summary"][key]
    m = s["notification_matching"]["5"]
    lines.append(f"Across {s['countries']} available countries: {s['raw']} raw anomalies, "
                 f"{s['coalesced']} coalesced, {s['retained']} impact-retained, "
                 f"{m['total']} notifications, {m['matched']} label-matched. "
                 f"Country-label coverage: {s['raw_covered']}/{s['labels_total']} to "
                 f"{s['retained_covered']}/{s['labels_total']}.")
    lines += ["", "## Country/global attribution conflicts", ""]
    for case in analysis["ordering_cases"]:
        lines.append(f"- {case['variant']}: {case['country']} {case['start']}: " + "; ".join(
            f"{kind}: {item['label']['date']} {item['label'].get('description', '')} ({item['distance']} days)"
            for kind in ["country_labels", "global_labels"] for item in case[kind]))
    lines += ["", "analysis.json includes the configuration comparisons, per-country calibration evidence, "
              "low-impact retained anomalies, lost labels, and notification groups. "
              "unmatched-notifications.csv uses all available label-constrained ensemble countries with "
              "knee/fallback selection and five-country grouping. Rows sharing a notification_id belong "
              "to one cross-country notification."]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-gap-days", type=int, default=1)
    parser.add_argument("--evaluation-labels", type=Path,
                        help="updated labels for test-period attribution; training labels must be unchanged")
    parser.add_argument("--all-configurations", action="store_true",
                        help="compare all four tuned configurations and the fixed baseline on common countries")
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "target/release/cenalert-detector",
                        help="current detector build used for contextualization")
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
    training_path = root / "inputs/training-labels.json"
    training_labels = read(training_path)
    if any(label["date"] > cutoff for label in training_labels):
        raise ValueError("training-label snapshot contains post-cutoff labels")
    test_labels = [l for l in labels if first <= l["date"] <= end]
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error("build the release binary or pass --binary")
    args.output.mkdir(parents=True, exist_ok=True)
    variants = VARIANTS if args.all_configurations else ["labels_ensemble"]
    runs = {(c, v): read(root / f"runs/{seed}/{c}/{v}/result.json")
            for c in config["countries"] for v in variants}
    common = common_completed_countries(runs, config["countries"], variants)
    if not common:
        raise ValueError("no countries completed every requested configuration")
    countries, excluded, ordering = [], [], []
    labels_by_country = {}
    summary, grouping, available_summary, available_grouping = {}, {}, {}, {}

    def key(variant, selection):
        return f"{variant}/{selection}" if args.all_configurations else selection

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for variant in variants:
            for country in config["countries"]:
                directory = root / f"runs/{seed}/{country}/{variant}"
                result = runs[country, variant]
                if result["status"] != "completed":
                    excluded.append(dict(country=country, variant=variant, status=result["status"]))
                    continue
                period_labels = [l for l in test_labels
                                 if result["test_first_date"] <= l["date"] <= result["test_last_date"]]
                if country in labels_by_country:
                    assert labels_by_country[country] == period_labels
                labels_by_country[country] = period_labels
                required = required_labels(period_labels, country, result["test_first_date"],
                                           result["test_last_date"], 0)
                train_required = required_labels(training_labels, country, "0001-01-01", cutoff, 0)
                for selection in (["fixed"] if variant == "fixed_3sigma" else ["knee", "leftmost"]):
                    detection = read(directory / f"{selection}-detection.json")
                    all_anomalies = detection["anomalies"]
                    training = [a for a in all_anomalies if a["end"] <= cutoff]
                    raw = [a for a in all_anomalies if first <= a["start"] <= end]
                    calibrated, threshold = filter_output(binary, country, training, training_path,
                                                         proximity, args.sparse_gap_days, scratch)
                    evidence = calibration_evidence(calibrated, training_labels, country, proximity, threshold)
                    coalesced, _ = filter_output(binary, country, raw, training_path,
                                                 proximity, args.sparse_gap_days, scratch)
                    retained = retain_by_impact(coalesced, threshold)
                    raw_covered = coverage(raw, required, proximity)
                    recorded = next(s for s in result["selections"] if s["selection"] == selection)
                    assert len(raw) == recorded["anomaly_count"]
                    if not args.evaluation_labels:
                        assert (len(required), len(raw_covered)) == (
                            recorded["country_labels_total"], recorded["country_labels_covered"])
                    retained_covered = coverage(retained, required, proximity)
                    assert coverage(coalesced, required, proximity) == raw_covered
                    countries.append(dict(country=country, variant=variant, selection=selection,
                        threshold=threshold, calibration=evidence, training_anomalies=len(training),
                        coalesced_training_anomalies=len(calibrated),
                        training_labels_total=len(train_required),
                        training_labels_covered=len(coverage([a for a in all_anomalies if a["start"] <= cutoff], train_required, proximity)),
                        raw=len(raw), raw_sparse=sum(a.get("sparse", False) for a in raw),
                        coalesced=len(coalesced), retained=len(retained),
                        labels_total=len(required), raw_covered=len(raw_covered),
                        retained_covered=len(retained_covered),
                        lost_labels=[l for l in raw_covered if l not in retained_covered],
                        excluded_boundary_intervals=[a for a in all_anomalies if a["start"] <= cutoff < a["end"]],
                        parameters=detection.get("parameters"), raw_anomalies=raw,
                        coalesced_anomalies=coalesced, retained_anomalies=retained))
                    if selection in ("knee", "fixed"):
                        ordering.extend(dict(variant=variant, **case)
                                        for case in ordering_cases(raw, period_labels, country, proximity))
            print(f"Reviewed {variant}", flush=True)
            for selection in (["fixed"] if variant == "fixed_3sigma" else ["knee", "leftmost"]):
                rows = [r for r in countries if r["variant"] == variant and r["selection"] == selection]
                name = key(variant, selection)
                available_summary[name], available_grouping[name] = aggregate_rows(
                    rows, binary, labels_by_country, proximity, scratch)
                selected = [r for r in rows if r["country"] in common]
                if len(selected) == len(rows):
                    summary[name], grouping[name] = available_summary[name], available_grouping[name]
                else:
                    summary[name], grouping[name] = aggregate_rows(
                        selected, binary, labels_by_country, proximity, scratch)
    analysis = dict(source=str(root), train_end=cutoff, test_end=end, seed=seed,
                    contextualization_binary=str(binary), country_proximity_days=proximity,
                    global_proximity_days=min(proximity, 1),
                    sparse_gap_days=args.sparse_gap_days, threshold_reference="training",
                    all_configurations=args.all_configurations, comparison_countries=common,
                    summary=summary, available_summary=available_summary, excluded=excluded,
                    countries=countries, ordering_cases=ordering, grouping=grouping,
                    available_grouping=available_grouping, evaluation_labels=labels,
                    low_impact_retained=low_impact_rows(countries, common))
    write(args.output / "analysis.json", analysis)
    write_unmatched_notifications(args.output / "unmatched-notifications.csv",
                                  available_grouping[key("labels_ensemble", "knee")]["5"])
    write_low_impact_review(args.output / "low-impact-review.csv", analysis)
    write_calibration_review(args.output / "low-calibration-countries.csv", countries)
    if args.all_configurations:
        write_paper_table(args.output / "contextualization-table.tex", available_summary, excluded)
        write_detection_table(args.output / "detection-table.tex", available_summary, excluded)
        write_configuration_summary(args.output / "configuration-summary.csv", summary)
        write_configuration_summary(args.output / "available-configuration-summary.csv", available_summary)
    (args.output / "report.md").write_text("\n".join(report_lines(analysis)) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
