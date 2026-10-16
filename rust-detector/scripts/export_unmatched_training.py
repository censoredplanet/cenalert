"""Export unlabelled coalesced training anomalies from a saved experiment."""
import argparse
import csv
import tempfile
from pathlib import Path

from batch_ensemble_tuning import label_applies
from review_contextualization import filter_output, matches, read, starts, write


def unmatched(anomalies, labels, country, proximity):
    applicable = [label for label in labels if label_applies(label, country)]
    return [a for a in anomalies if not any(matches(a, l, proximity) for l in applicable)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection", choices=["knee", "leftmost"], default="knee")
    parser.add_argument("--sparse-gap-days", type=int, default=1)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] /
                        "target/release/cenalert-detector")
    args = parser.parse_args()
    config = read(args.results / "experiment.json")
    if len(config["seeds"]) != 1 or args.sparse_gap_days < 0:
        parser.error("requires one experiment seed and a nonnegative sparse gap")
    cutoff = config["train_end"]
    labels = [l for l in read(args.labels) if l["date"] <= cutoff]
    rows, completed, skipped = [], [], []
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        labels_path = scratch / "labels.json"
        write(labels_path, labels)
        for country in config["countries"]:
            directory = args.results / f"runs/{config['seeds'][0]}/{country}/labels_ensemble"
            result = read(directory / "result.json")
            if result["status"] != "completed":
                skipped.append(country)
                continue
            anomalies = read(directory / f"{args.selection}-detection.json")["anomalies"]
            training = [a for a in anomalies if a["end"] <= cutoff]
            coalesced, _ = filter_output(args.binary.resolve(), country, training, labels_path,
                                        config["proximity_days"], args.sparse_gap_days, scratch)
            completed.append(country)
            for a in unmatched(coalesced, labels, country, config["proximity_days"]):
                rows.append(dict(country=country, start=a["start"], end=a["end"],
                                 peak=a["peak"], impact=a["impact"], sparse=a.get("sparse", False),
                                 component_starts="; ".join(starts(a))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["country", "start", "end", "peak",
                                                   "impact", "sparse", "component_starts"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} unmatched training anomalies across {len(completed)} completed "
          f"countries (labels_ensemble/{args.selection}, through {cutoff}, sparse gap "
          f"{args.sparse_gap_days}). Skipped: {', '.join(skipped) or 'none'}.")


if __name__ == "__main__":
    main()
