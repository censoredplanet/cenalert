"""Append reviewed event sheets to an existing private label file."""
import argparse
import json
from pathlib import Path

from batch_ensemble_tuning import label_country_code
from build_labels import normalize_event, read_additional_events


def identity(label):
    return (label["scope"], label_country_code(label) or "", label["date"])


def merge_updates(labels, paths):
    merged = list(labels)
    added = []
    duplicates = 0
    for path in paths:
        for row in read_additional_events(path):
            row["source"] = row.get("source", "").strip() or path.name
            scope = "country" if row["country_code"].strip() else "global"
            event = normalize_event(row, scope, "additional", "manual")
            same_day = [label for label in merged if identity(label) == identity(event)]
            if any((label.get("description") or "").strip() == (event.get("description") or "").strip()
                   for label in same_day):
                duplicates += 1
                continue
            if same_day:
                raise ValueError(f"review same-day event before merging: {identity(event)}: "
                                 f"{event.get('description', '')}")
            merged.append(event)
            added.append(event)
    return merged, added, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--additional-events", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels:
        parser.error("labels must be a nonempty array")
    merged, added, duplicates = merge_updates(labels, args.additional_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Preserved {len(labels)} existing labels; added {len(added)}; "
          f"skipped {duplicates} duplicates; total {len(merged)}.")
    for label in added:
        print(f"{label['date']} {label.get('country_code', 'Global')}: {label.get('description', '')}")


if __name__ == "__main__":
    main()
