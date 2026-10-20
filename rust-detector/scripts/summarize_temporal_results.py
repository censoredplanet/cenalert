"""Summarize saved CenAlert temporal experiments for manuscript discussion."""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from batch_ensemble_tuning import matched_label
from review_contextualization import evaluation_labels

GROUPS = ["original", "labels", "ensemble", "labels_ensemble"]
NAMES = {"original": "Chebyshev", "labels": "Chebyshev + labels",
         "ensemble": "Ensemble", "labels_ensemble": "Ensemble + labels",
         "fixed_3sigma": "Fixed 3-sigma"}
MEMBERS = ["chebyshev", "isolation_forest", "lof"]
COLORS = ["#245a81", "#248375", "#c47726", "#854b86", "#777777"]


def read_json(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def reevaluate_test_selections(index, runs, labels, proximity, evaluate_period):
    updated = {}
    for (country, seed, variant, selection), saved in index.items():
        run = runs[country, seed, variant]
        period = evaluate_period(saved["anomalies"], labels, country,
                                 run["test_first_date"], run["test_last_date"], proximity)
        assert period["anomaly_count"] == saved["anomaly_count"]
        updated[country, seed, variant, selection] = dict(saved, **period)
    return updated


def selected_point(directory, selection, variant, front):
    parameters = read_json(directory / f"{selection}-parameters.json")
    fields = (["window", "min_residual_ratio", "efficiency", "chebyshev_z", "chebyshev_k",
               "isolation_forest_min_score", "lof_min_score"] if "ensemble" in variant else
              ["window", "z", "k", "min_residual_ratio", "efficiency"])
    candidates = []
    for values, objective in front:
        resolved = [math.floor(values[0] + .5), *values[1:]]
        if all(math.isclose(resolved[i], parameters[field], rel_tol=1e-10, abs_tol=1e-10)
               for i, field in enumerate(fields)):
            candidates.append(objective)
    if not candidates:
        raise ValueError(f"Cannot locate saved selection: {directory}/{selection}")
    assert all(np.allclose(item, candidates[0]) for item in candidates)
    return [candidates[0][0], -candidates[0][1]]


def draw_front(ax, info, label=True):
    points = info["points"]
    ax.plot([p[0] for p in points], [p[1] for p in points], ".-", color=COLORS[0],
            linewidth=1, markersize=3, label="Saved Pareto front")
    for selection, marker, color in [("knee", "o", COLORS[2]), ("leftmost", "x", COLORS[3])]:
        x, y = info[selection]
        name = ("Knee" if info["knee_found"] else "Leftmost fallback") if selection == "knee" else "Leftmost"
        ax.scatter([x], [y], marker=marker, s=55 if selection == "knee" else 45,
                   facecolors="none" if selection == "knee" else color,
                   edgecolors=color if selection == "knee" else None, color=color,
                   linewidths=1.7, zorder=5, label=f"{name}: N={int(x)}")
    ax.set_xlabel("Training anomalies")
    ax.set_ylabel("Total training impact")
    ax.grid(alpha=.18)
    if label:
        ax.legend(fontsize=7, loc="best")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pareto-countries", nargs="+", default=["TR", "OM"],
                        help="countries shown in the manuscript Pareto figure")
    parser.add_argument("--evaluation-labels", type=Path,
                        help="updated test labels; training labels must be unchanged")
    args = parser.parse_args()
    source, output = args.results.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(source / "experiment.json")
    if len(config["seeds"]) != 1:
        raise ValueError("This report requires a single-seed experiment")
    sys.path.insert(0, str(source / "inputs"))
    from run_experiment_matrix import evaluate_period
    labels = read_json(source / "inputs/labels.json")
    if args.evaluation_labels:
        labels = evaluation_labels(labels, read_json(args.evaluation_labels), config["train_end"])
    summary = list(csv.DictReader((source / "summary.csv").open()))
    runs = {(r["country"], r["seed"], r["variant"]): r
            for p in sorted((source / "runs").glob("*/*/*/result.json")) for r in [read_json(p)]}
    expected = {(c, seed, variant) for c in config["countries"]
                for seed in config["seeds"] for variant in config["variants"]}
    if set(runs) != expected:
        raise ValueError("Missing or unexpected runs; finish the experiment before summarizing")
    index = {(c, seed, variant, s["selection"]): s for (c, seed, variant), r in runs.items()
             for s in r.get("selections", [])}
    # Check the aggregate CSV against the saved per-run records before aggregation.
    for row in summary:
        if row["status"] != "completed":
            continue
        s = index[(row["country"], int(row["seed"]), row["variant"], row["selection"])]
        for field in ["anomaly_count", "country_labels_total", "country_labels_covered", "unmatched_anomalies"]:
            assert int(row[field]) == s[field], (row, field)
        assert len(s["anomalies"]) == s["anomaly_count"]
        assert sum(bool(x["matched_starts"]) for x in s["labels"]) == s["country_labels_covered"]

    if args.evaluation_labels:
        index = reevaluate_test_selections(index, runs, labels, config["proximity_days"], evaluate_period)

    common = set.intersection(*[{(c, seed) for (c, seed, v), r in runs.items()
                                if v == variant and r["status"] == "completed"}
                               for variant in GROUPS + ["fixed_3sigma"]])
    matrix = []
    for variant in GROUPS + ["fixed_3sigma"]:
        for selection in (["fixed"] if variant == "fixed_3sigma" else ["knee", "leftmost"]):
            selected = [index[c, seed, variant, selection] for c, seed in sorted(common)]
            record = {"variant": variant, "selection": selection, "country_seed_pairs": len(common)}
            for field in ["anomaly_count", "country_labels_total", "country_labels_covered", "unmatched_anomalies"]:
                record[field] = sum(s[field] for s in selected)
            record["label_matched_anomalies"] = record["anomaly_count"] - record["unmatched_anomalies"]
            training = Counter()
            for c, seed in sorted(common):
                detection = read_json(source / f"runs/{seed}/{c}/{variant}/{selection}-detection.json")
                period = evaluate_period(detection["anomalies"], labels, c, "0001-01-01",
                                         config["train_end"], config["proximity_days"])
                training.update({key: period[key] for key in
                                 ["anomaly_count", "country_labels_total", "country_labels_covered"]})
            record["training"] = dict(training)
            matrix.append(record)
    status = {v: dict(Counter(r["status"] for (c, seed, g), r in runs.items() if g == v))
              for v in GROUPS + ["fixed_3sigma"]}
    failures = [{"country": c, "seed": seed, "variant": v, "status": r["status"]}
                for (c, seed, v), r in sorted(runs.items()) if r["status"] != "completed"]
    coverage = read_json(source / "input-summary.json")
    test_start = min(coverage[c]["test_first_date"] for c, seed in common)
    test_end = max(coverage[c]["test_last_date"] for c, seed in common)

    votes, vote_country = [], []
    for variant in ["ensemble", "labels_ensemble"]:
        for selection in ["knee", "leftmost"]:
            totals = Counter()
            scores = []
            thresholds = []
            for c, seed in sorted(common):
                s = index[c, seed, variant, selection]
                local = Counter()
                thresholds.append(read_json(source / f"runs/{seed}/{c}/{variant}/{selection}-parameters.json")["lof_min_score"])
                for anomaly in s["anomalies"]:
                    if anomaly.get("sparse", False):
                        local["croston"] += 1
                        continue
                    vote = anomaly.get("ensemble_vote")
                    if not vote:
                        raise ValueError(f"Missing dense member votes: {c}/{variant}/{selection}")
                    yes = [name for name in MEMBERS if vote[name]["voted"]]
                    assert len(yes) == vote["vote_count"] and len(yes) >= 2
                    local["dense"] += 1
                    local["+".join(yes)] += 1
                    for name in yes:
                        local[name] += 1
                    if "lof" in yes and len(yes) == 2:
                        local["lof_decisive"] += 1
                    if vote["lof"]["score"] is not None:
                        scores.append(vote["lof"]["score"])
                vote_country.append(dict(country=c, seed=seed, variant=variant, selection=selection, **local))
                totals.update(local)
            votes.append(dict(variant=variant, selection=selection, **totals,
                lof_score_quantiles=dict(zip(["minimum", "median", "p90", "maximum"],
                                             map(float, np.quantile(scores, [0, .5, .9, 1])))),
                lof_threshold_median=float(np.median(thresholds)),
                lof_threshold_above_3=sum(t > 3 for t in thresholds)))

    differences = []
    for row in csv.DictReader((source / "detector-differences.csv").open()):
        if (row["country"], int(row["seed"])) not in common:
            continue
        if args.evaluation_labels:
            run = runs[row["country"], int(row["seed"]), row["variant"]]
            period_labels = [l for l in labels
                             if run["test_first_date"] <= l["date"] <= run["test_last_date"]]
            match = matched_label(row, period_labels, row["country"], config["proximity_days"])
            row["label_match"] = json.dumps(match)
        else:
            match = json.loads(row["label_match"])
        differences.append(dict(row, matched=match is not None))
    diff_totals = []
    for variant in ["ensemble", "labels_ensemble"]:
        for selection in ["knee", "leftmost"]:
            for direction in ["ensemble_only", "chebyshev_only"]:
                rows = [r for r in differences if (r["variant"], r["selection"], r["direction"]) ==
                        (variant, selection, direction)]
                diff_totals.append(dict(variant=variant, selection=selection, direction=direction,
                                       anomalies=len(rows), matched=sum(r["matched"] for r in rows)))
    # Review queue uses one row per country/seed/start and includes every selection that found it.
    queue = {}
    for r in differences:
        if r["direction"] != "ensemble_only" or r["matched"]:
            continue
        key = (r["country"], r["seed"], r["start"])
        record = queue.setdefault(key, dict(country=r["country"], seed=r["seed"], start=r["start"],
                                            configurations=[], intervals=[], impacts=[]))
        record["configurations"].append(r["variant"] + "/" + r["selection"])
        record["intervals"].append(r["start"] + "/" + r["end"])
        record["impacts"].append(float(r["impact"]))

    snapshots = {}
    for name, items in [("experiment", labels),
                        ("training", read_json(source / "inputs/training-labels.json"))]:
        counts = Counter((x["scope"], str(x.get("metadata", {}).get("category", "uncategorized"))) for x in items)
        snapshots[name] = dict(total=len(items), scope=dict(Counter(x["scope"] for x in items)),
            categories=[dict(scope=scope, category=category, count=count)
                        for (scope, category), count in sorted(counts.items())])

    fronts = []
    for (c, seed, variant), r in sorted(runs.items()):
        if variant not in GROUPS or r["status"] != "completed":
            continue
        directory = source / f"runs/{seed}/{c}/{variant}"
        front = read_json(directory / "pareto-front.json")
        by_count = {}
        for values, objective in front:
            by_count[objective[0]] = max(by_count.get(objective[0], -math.inf), -objective[1])
        knee = selected_point(directory, "knee", variant, front)
        left = selected_point(directory, "leftmost", variant, front)
        fronts.append(dict(country=c, seed=seed, variant=variant, points=sorted(by_count.items()),
            knee=knee, leftmost=left, knee_found=index[c, seed, variant, "knee"]["knee_found"],
            test_knee=index[c, seed, variant, "knee"]["anomaly_count"],
            test_leftmost=index[c, seed, variant, "leftmost"]["anomaly_count"],
            training_labels=sum(x["scope"] == "country" and x.get("country_code") == c
                                for x in read_json(source / "inputs/training-labels.json"))))
    front_index = {(x["country"], x["seed"], x["variant"]): x for x in fronts}

    metrics = dict(source=str(source), configuration=config, test_start=test_start, test_end=test_end,
        common_country_seed_pairs=sorted(common), status=status, failures=failures, matrix=matrix,
        votes=votes, votes_by_country=vote_country, differences=diff_totals,
        review_queue=list(queue.values()), labels=snapshots, evaluation_labels=labels, fronts=fronts,
        pareto_countries=args.pareto_countries)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), layout="constrained")
    x = np.arange(5)
    for selection, offset, color in [("knee", -.18, COLORS[0]), ("leftmost", .18, COLORS[2])]:
        records = [next(r for r in matrix if r["variant"] == v and r["selection"] ==
                        ("fixed" if v == "fixed_3sigma" else selection)) for v in GROUPS + ["fixed_3sigma"]]
        for ax, field in zip(axes, ["anomaly_count", "country_labels_covered"]):
            bars = ax.bar(x[:4] + offset, [r[field] for r in records[:4]], width=.34, color=color,
                          label="Knee / fallback" if selection == "knee" else "Leftmost")
            ax.bar_label(bars, fontsize=7, padding=2)
    for ax, field, title in zip(axes, ["anomaly_count", "country_labels_covered"],
                               ["Test anomalies", f"Held-out country labels covered ({matrix[0]['country_labels_total']} total)"]):
        fixed = next(r for r in matrix if r["variant"] == "fixed_3sigma")
        bars = ax.bar([4], [fixed[field]], width=.5, color=COLORS[4], label="Fixed baseline")
        ax.bar_label(bars, fontsize=7, padding=2)
        ax.set_xticks(x, ["Cheb.", "Cheb.\n+ labels", "Ensemble", "Ensemble\n+ labels", "Fixed 3σ"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(r[field] for r in matrix) * 1.18)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=7, loc="upper left")
    save_figure(fig, output, "evaluation-matrix")

    fig, ax = plt.subplots(figsize=(7.4, 3.0), layout="constrained")
    patterns = ["chebyshev+isolation_forest", "chebyshev+lof", "isolation_forest+lof",
                "chebyshev+isolation_forest+lof"]
    bottoms = np.zeros(4)
    for pattern, color, label in zip(patterns, COLORS,
            ["Cheb. + IF", "Cheb. + LOF", "IF + LOF", "All three"]):
        values = np.array([r.get(pattern, 0) for r in votes])
        ax.barh(np.arange(4), values, left=bottoms, label=label, color=color)
        for i, count in enumerate(values):
            if count >= 5:
                ax.text(bottoms[i] + count / 2, i, str(count), ha="center", va="center", color="white", fontsize=8)
        bottoms += values
    ax.set_yticks(range(4), ["Ensemble: knee", "Ensemble: leftmost", "Ensemble + labels: knee", "Ensemble + labels: leftmost"])
    ax.invert_yaxis()
    ax.set_xlabel("Dense test anomaly onsets")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(.5, 1.22), fontsize=8)
    save_figure(fig, output, "ensemble-votes")

    seed = config["seeds"][0]
    examples = args.pareto_countries
    missing = [c for c in examples if (c, seed, "labels_ensemble") not in front_index]
    if missing:
        raise ValueError(f"No completed label-constrained ensemble front for: {', '.join(missing)}")
    fig, axes = plt.subplots(1, len(examples), squeeze=False,
                             figsize=(4.0 * len(examples), 2.8), layout="constrained")
    for ax, c in zip(axes.flat, examples):
        info = front_index[c, seed, "labels_ensemble"]
        draw_front(ax, info)
        ax.set_title(f"{c}: ensemble + labels", fontsize=10)
    save_figure(fig, output, "pareto-examples")

    with (output / "unlabeled-ensemble-detections.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["country", "seed", "start", "configurations", "intervals", "impacts"])
        writer.writeheader()
        for record in sorted(queue.values(), key=lambda r: (r["country"], r["start"])):
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in record.items()})

    with PdfPages(output / "pareto-all-countries.pdf") as pdf:
        for c, seed in sorted({(c, seed) for c, seed, v in runs}):
            fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.2), layout="constrained")
            fig.suptitle(f"{c} · seed {seed} · training through {config['train_end']}", fontsize=13)
            for ax, v in zip(axes.flat, GROUPS):
                info = front_index.get((c, seed, v))
                if info is None:
                    ax.text(.5, .5, runs[c, seed, v]["status"].replace("_", " "),
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                else:
                    draw_front(ax, info)
                subtitle = (f"\nTest anomalies: knee {info['test_knee']}, left {info['test_leftmost']}" if info else "")
                ax.set_title(NAMES[v] + subtitle, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)

    missed = [{"country": c, **item["label"]}
              for c, run_seed in sorted(common)
              for item in index[c, run_seed, "labels_ensemble", "knee"]["labels"]
              if not item["matched_starts"]]
    metrics["missed_test_labels"] = missed
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    write_reports(output, metrics, front_index, seed)
    print(json.dumps({"output": str(output), "matrix": matrix, "votes": votes,
                      "differences": diff_totals, "review_starts": len(queue)}, indent=2))


def write_reports(output, m, fronts, seed):
    config = m["configuration"]
    n = len(m["common_country_seed_pairs"])
    by_key = {(r["variant"], r["selection"]): r for r in m["matrix"]}
    plain = by_key["ensemble", "knee"]
    labeled = by_key["labels_ensemble", "knee"]
    left = by_key["labels_ensemble", "leftmost"]
    baseline = by_key["fixed_3sigma", "fixed"]
    total = labeled["country_labels_total"]
    lof = next(r for r in m["votes"] if (r["variant"], r["selection"]) == ("labels_ensemble", "knee"))
    skipped = ", ".join(sorted({r["country"] for r in m["failures"] if r["status"].startswith("skipped")}))
    rows = []
    for r in m["matrix"]:
        t = r["training"]
        rows.append([NAMES[r["variant"]], r["selection"], t["anomaly_count"],
                     f"{t['country_labels_covered']}/{t['country_labels_total']}",
                     r["anomaly_count"], f"{r['country_labels_covered']}/{total}",
                     r["label_matched_anomalies"], r["unmatched_anomalies"]])
    matrix_table = table(["Configuration", "Selection", "Train anomalies", "Train labels",
                          "Test anomalies", "Test labels", "Matched test anomalies", "Unmatched"], rows)
    vote_table = table(["Configuration", "Selection", "Dense", "Sparse", "Cheb+IF", "Cheb+LOF", "IF+LOF", "All three"],
                       [[NAMES[r["variant"]], r["selection"], r.get("dense", 0), r.get("croston", 0),
                         r.get("chebyshev+isolation_forest", 0), r.get("chebyshev+lof", 0),
                         r.get("isolation_forest+lof", 0), r.get("chebyshev+isolation_forest+lof", 0)] for r in m["votes"]])
    missing_table = table(["Country", "Date", "Event"],
                          [[r["country"], r["date"], r.get("description", "")] for r in m["missed_test_labels"]])
    status_table = table(["Country", "Configuration", "Status"],
                         [[r["country"], NAMES[r["variant"]], r["status"]] for r in m["failures"]])
    comparison = (f"At knee/fallback selection, the ensemble detects {plain['anomaly_count']} test anomalies and covers "
                  f"{plain['country_labels_covered']}/{total} country labels. With training-label constraints, it detects "
                  f"{labeled['anomaly_count']} anomalies and covers {labeled['country_labels_covered']}/{total}. "
                  f"The fixed 3-sigma baseline detects {baseline['anomaly_count']} anomalies and covers "
                  f"{baseline['country_labels_covered']}/{total} labels.")
    selection = (f"For the labeled ensemble, leftmost selection detects {left['anomaly_count']} test anomalies and covers "
                 f"{left['country_labels_covered']}/{total} labels, compared with {labeled['anomaly_count']} and "
                 f"{labeled['country_labels_covered']}/{total} for knee/fallback selection.")
    members = (f"The labeled ensemble with knee/fallback selection produces {lof.get('dense', 0)} dense and "
               f"{lof.get('croston', 0)} sparse test anomalies. At dense anomaly onsets, "
               f"{lof.get('chebyshev+isolation_forest', 0)} receive votes from Chebyshev and Isolation Forest, "
               f"{lof.get('chebyshev+lof', 0)} from Chebyshev and LOF, "
               f"{lof.get('isolation_forest+lof', 0)} from Isolation Forest and LOF, and "
               f"{lof.get('chebyshev+isolation_forest+lof', 0)} from all three. "
               f"LOF supplies the second vote required for a majority at {lof.get('lof_decisive', 0)} onsets. "
               "Sparse anomalies use the shared Croston path.")
    setup = (f"Training ends on {config['train_end']}. Test observations span {m['test_start']} to {m['test_end']}. "
             f"Each tuned run uses {config['evaluations']:,} evaluations and population {config['population_size']}. "
             f"The comparison includes {n} countries completed by all five configurations, with seed {seed}. "
             "The optimizer's initial population is unseeded. "
             f"Country-label matching uses a ±{config['proximity_days']}-day window. "
             "Counts describe raw detections before sparse coalescing, impact filtering, and cross-country grouping. "
             "Matched anomalies include country and global labels. A single anomaly may match multiple labels.")
    inventory = (f"The evaluation label set contains {m['labels']['experiment']['total']} labels: "
                 f"{m['labels']['experiment']['scope'].get('country', 0)} country and "
                 f"{m['labels']['experiment']['scope'].get('global', 0)} global. "
                 f"Of these, {m['labels']['training']['total']} are dated through the training cutoff.")
    text = ["# CenAlert temporal evaluation", "", setup, "", "## Evaluation matrix", "", matrix_table, "", comparison,
            "", "![Evaluation matrix](evaluation-matrix.png)", "", "## Parameter selection", "", selection,
            "", "![Training Pareto fronts](pareto-examples.png)", "",
            f"Examples: {', '.join(m['pareto_countries'])}. Markers locate the saved selected parameters on the training fronts.",
            "", "[All-country fronts](pareto-all-countries.pdf)", "", "## Ensemble members", "", members, "", vote_table,
            "", "![Member votes](ensemble-votes.png)", "", "## Labels and review", "", inventory, "", missing_table,
            "", f"The unlabeled ensemble-only review queue contains {len(m['review_queue'])} distinct country/seed/start records. "
            "It compares ensemble and single-detector outputs with the same constraint setting and selection rule, using ±3-day start proximity. "
            "The records are in `unlabeled-ensemble-detections.csv`.", "",
            "Labels are incomplete, so unmatched detections cannot be counted as false positives.",
            "", "## Infeasible and skipped runs", "", status_table, "",
            "No feasible solution means that the search budget found no parameters preserving all training labels. "
            "It does not establish that no such parameters exist.", "", "## Overleaf", "",
            "Use the generated table and statistics to update the corresponding manuscript passages. "
            "The Pareto examples figure is available for the manuscript. "
            "The preamble needs graphicx and booktabs."]
    (output / "results.md").write_text("\n".join(text) + "\n")
    zh = ["# CenAlert 本轮实验结果", "",
          f"训练截至 {config['train_end']}，测试期为 {m['test_start']} 至 {m['test_end']}。每组调参 {config['evaluations']:,} 次，种群 {config['population_size']}，种子 {seed}。",
          f"主表统一使用 {n} 个国家。各配置未完成的国家见下方明细；{skipped} 因训练数据全零跳过。",
          "", "## 四组对照与固定基线", "", matrix_table, "",
          f"Ensemble 的 knee/fallback 输出为 {plain['anomaly_count']} 个异常、覆盖 {plain['country_labels_covered']}/{total} 个测试标签；加入训练标签约束后为 {labeled['anomaly_count']} 个异常、覆盖 {labeled['country_labels_covered']}/{total}。",
          f"固定 3σ 产生 {baseline['anomaly_count']} 个异常，覆盖 {baseline['country_labels_covered']}/{total}。",
          "以上均为过滤前的异常。Matched test anomalies 是匹配到国家或全球标签的异常数量，一个异常可能匹配多个标签。",
          "", "![四组对照](evaluation-matrix.png)", "", "## 参数选择", "",
          f"带标签 ensemble 使用 leftmost 得到 {left['anomaly_count']} 个异常、覆盖 {left['country_labels_covered']}/{total}，使用 knee/fallback 得到 {labeled['anomaly_count']} 个异常、覆盖 {labeled['country_labels_covered']}/{total}。",
          "", "![Pareto 曲线](pareto-examples.png)", "", f"示例国家：{', '.join(m['pareto_countries'])}。全部国家的四组曲线见 pareto-all-countries.pdf。",
          "", "## 成员投票", "", vote_table, "",
          f"带标签 ensemble 的 knee/fallback 输出包括 {lof.get('dense', 0)} 个稠密异常和 {lof.get('croston', 0)} 个稀疏异常。LOF 在 {lof.get('lof_decisive', 0)} 个稠密异常起点提供凑足多数票的第二票。",
          "", "![投票分布](ensemble-votes.png)", "", "## 未覆盖的测试标签", "", missing_table, "",
          f"标签快照共 {m['labels']['experiment']['total']} 条，其中 {m['labels']['training']['total']} 条日期位于训练期。",
          f"ensemble 独有且未匹配标签的核查队列有 {len(m['review_queue'])} 个国家／种子／开始日期记录，见 unlabeled-ensemble-detections.csv。标签不完整，未匹配异常不能直接作为误报。",
          "", "## 未找到可行解与跳过项", "", status_table, "",
          "未找到可行解表示在本次搜索预算内无法保留全部训练标签。具体缺失标签需要另做诊断运行，失败任务没有保存候选解。",
          "", "## 更新论文", "", "根据生成的表格和统计更新论文对应段落。Pareto 示例图可用于正文。"]
    (output / "results-zh.md").write_text("\n".join(zh) + "\n")
    failure_text = " ".join(
        f"{NAMES[v]} found no feasible solution for " +
        ", ".join(sorted(r['country'] for r in m['failures']
                        if r['variant'] == v and r['status'] == 'no_feasible_solution')) + "."
        for v in GROUPS if any(r['variant'] == v and r['status'] == 'no_feasible_solution'
                              for r in m['failures']))
    latex = [r"\section{Results}", r"\subsection{Experimental Setup}",
             f"We tune on observations through {config['train_end']} and evaluate on {m['test_start']}--{m['test_end']}. "
             f"Each tuned configuration uses {config['evaluations']:,} NSGA-II evaluations and a population of {config['population_size']}. "
             f"We report one run with seed {seed}; the initial population is unseeded. "
             rf"Table~\ref{{tab:cenalert-matrix}} compares {n} countries completed by all configurations. "
             f"{failure_text} "
             f"{skipped} has an all-zero training series. "
             "The counts precede sparse coalescing, impact filtering, and cross-country notification grouping. "
             "Label coverage uses country events and a start-date proximity window of three days. "
             "Matched anomaly counts also include global events. Labels provide incomplete event coverage.",
             r"\begin{table*}[t]", r"\centering\small",
             rf"\caption{{Detection and label coverage across {n} common countries. K: knee with leftmost fallback. L: leftmost. Matched counts test anomalies with a country or global label.}}",
             r"\label{tab:cenalert-matrix}", r"\begin{tabular}{llrrrrr}\toprule",
             r"Configuration & Rule & Train anomalies & Train labels & Test anomalies & Test labels & Matched \\\midrule"]
    for r in m["matrix"]:
        t = r["training"]
        name = NAMES[r["variant"]].replace("Fixed 3-sigma", r"Fixed $3\sigma$")
        rule = {"knee": "K", "leftmost": "L", "fixed": "--"}[r["selection"]]
        latex.append(f"{name} & {rule} & {t['anomaly_count']} & {t['country_labels_covered']}/{t['country_labels_total']} & "
                     f"{r['anomaly_count']} & {r['country_labels_covered']}/{total} & {r['label_matched_anomalies']} " + r"\\")
    latex += [r"\bottomrule\end{tabular}", r"\end{table*}", r"\subsection{Detection and Label Coverage}", comparison,
              "Unmatched detections remain candidates for investigation. These partial labels do not establish overall precision or recall.",
              r"\subsection{Parameter Selection}", selection,
              r"Figure~\ref{fig:pareto-selection} shows training fronts and selected solutions for " + ", ".join(m['pareto_countries']) + ".",
              r"\begin{figure*}[t]", r"\centering",
              r"\includegraphics[width=0.85\textwidth]{figures/pareto-examples.pdf}",
              r"\caption{Training Pareto fronts for label-constrained ensemble tuning. The axes show anomaly count and total impact. Markers identify the selected knee or fallback and the leftmost solution.}",
              r"\label{fig:pareto-selection}", r"\end{figure*}",
              r"\subsection{Ensemble Member Contributions}", members]
    (output / "results.tex").write_text("\n".join(latex) + "\n")


if __name__ == "__main__":
    main()
