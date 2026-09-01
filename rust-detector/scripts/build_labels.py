#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


COUNTRY_SHEETS = ("filtering", "shutdowns", "other")
EVENT_FIELDS = (
    "country",
    "country_code",
    "affected_regions",
    "start_date",
    "end_date",
    "cause",
    "affected_services",
    "link",
    "source",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CenAlert labels from manually and automatically verified events."
    )
    for category in COUNTRY_SHEETS:
        parser.add_argument(f"--{category}", required=True, type=Path)
    parser.add_argument("--global-events", required=True, type=Path)
    parser.add_argument(
        "--additional-events",
        action="append",
        default=[],
        type=Path,
        help="Optional mixed country/global event CSV; may omit the standard header.",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="Optional CSV of country/date/description/category labels to omit.",
    )
    parser.add_argument(
        "--cross-country-matches",
        type=Path,
        help="Optional CSV assigning country events to explicit additional countries.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--community-events", type=Path)
    parser.add_argument("--series-directory", type=Path)
    parser.add_argument("--parameters-directory", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--proximity-days", type=int, default=3)
    arguments = parser.parse_args()

    automatic = (
        arguments.community_events,
        arguments.series_directory,
        arguments.parameters_directory,
        arguments.detector,
    )
    if any(automatic) and not all(automatic):
        parser.error(
            "automatic matching requires --community-events, --series-directory, "
            "--parameters-directory, and --detector"
        )
    if arguments.proximity_days < 0:
        parser.error("--proximity-days must be non-negative")
    return arguments


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as event_file:
        rows = list(csv.DictReader(event_file))
    required = {"country_code", "start_date", "affected_services", "source"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def read_additional_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as event_file:
        records = list(csv.reader(event_file))
    if not records:
        return []
    if tuple(value.strip() for value in records[0]) == EVENT_FIELDS:
        records = records[1:]
    rows = []
    for row_number, values in enumerate(records, start=1):
        if len(values) != len(EVENT_FIELDS):
            raise ValueError(
                f"{path}:{row_number} has {len(values)} columns; "
                f"expected {len(EVENT_FIELDS)}"
            )
        rows.append(dict(zip(EVENT_FIELDS, values)))
    return rows


def normalize_event(
    row: dict[str, str], scope: str, category: str, verification: str
) -> dict[str, Any]:
    start_date = row["start_date"].strip()
    date.fromisoformat(start_date)
    country_code = row.get("country_code", "").strip().upper()
    if scope == "country" and not country_code:
        raise ValueError(f"country event on {start_date} has no country_code")

    description = row.get("affected_services", "").strip() or row.get("cause", "").strip()
    metadata = {
        key: value.strip()
        for key, value in row.items()
        if key not in {"start_date", "country_code"} and value and value.strip()
    }
    metadata["category"] = category
    metadata["verification"] = verification

    label: dict[str, Any] = {
        "date": start_date,
        "scope": scope,
        "description": description or None,
        "metadata": metadata,
    }
    if country_code:
        label["country_code"] = country_code
    return {key: value for key, value in label.items() if value is not None}


def load_manual_labels(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for category in COUNTRY_SHEETS:
        for row in read_events(getattr(arguments, category)):
            labels.append(normalize_event(row, "country", category, "manual"))
    for row in read_events(arguments.global_events):
        labels.append(normalize_event(row, "global", "global", "manual"))
    for path in arguments.additional_events:
        for row in read_additional_events(path):
            scope = "country" if row["country_code"].strip() else "global"
            labels.append(normalize_event(row, scope, "additional", "manual"))
    return labels


def apply_cross_country_matches(
    labels: list[dict[str, Any]], rules_path: Path | None
) -> tuple[list[dict[str, Any]], int]:
    if rules_path is None:
        return labels, 0
    with rules_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "country_code",
            "start_date",
            "description",
            "additional_matching_countries",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{rules_path} is missing columns: {', '.join(sorted(missing))}"
            )
        rules = list(reader)

    applied = 0
    for rule in rules:
        country_code = rule["country_code"].strip().upper()
        start_date = rule["start_date"].strip()
        description = rule["description"].strip()
        date.fromisoformat(start_date)
        matches = [
            label
            for label in labels
            if label["scope"] == "country"
            and label.get("country_code", "").upper() == country_code
            and label["date"] == start_date
            and label.get("description", "") == description
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cross-country rule for {country_code} {start_date} matched "
                f"{len(matches)} labels; expected exactly one"
            )
        additional = sorted(
            {
                value.strip().upper()
                for value in rule["additional_matching_countries"].split(";")
                if value.strip()
            }
        )
        if not additional or any(
            len(country) != 2 or not country.isalpha() for country in additional
        ):
            raise ValueError(
                f"cross-country rule for {country_code} {start_date} has invalid "
                "additional country codes"
            )
        matches[0]["additional_matching_countries"] = additional
        reason = rule.get("reason", "").strip()
        if reason:
            matches[0]["metadata"]["cross_country_match_reason"] = reason
        applied += 1
    return labels, applied


def detect_anomaly_starts(
    detector: Path, series_path: Path, parameters_path: Path, country_code: str
) -> list[date]:
    with series_path.open(newline="", encoding="utf-8-sig") as series_file:
        observations = [
            {
                "country": country_code,
                "date": row["date"],
                "rate": float(row["value"]),
                "topic": "vpn",
            }
            for row in csv.DictReader(series_file)
        ]

    with tempfile.TemporaryDirectory(prefix="cenalert-labels-") as directory:
        dashboard_path = Path(directory) / f"{country_code}.json"
        dashboard_path.write_text(json.dumps(observations), encoding="utf-8")
        completed = subprocess.run(
            [
                str(detector),
                "detect",
                "--input",
                str(dashboard_path),
                "--parameters",
                str(parameters_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    output = json.loads(completed.stdout)
    return [date.fromisoformat(anomaly["start"]) for anomaly in output["anomalies"]]


def derive_automatic_labels(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    events_by_country: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_events(arguments.community_events):
        country_code = row["country_code"].strip().upper()
        if country_code:
            events_by_country[country_code].append(row)

    labels: list[dict[str, Any]] = []
    for country_code, events in sorted(events_by_country.items()):
        series_path = arguments.series_directory / f"{country_code}.csv"
        parameters_path = arguments.parameters_directory / f"{country_code}.json"
        if not series_path.exists() or not parameters_path.exists():
            print(
                f"warning: skipping automatic events for {country_code}; "
                "series or selected parameters are unavailable",
                file=sys.stderr,
            )
            continue

        anomaly_starts = detect_anomaly_starts(
            arguments.detector, series_path, parameters_path, country_code
        )
        for event in events:
            event_date = date.fromisoformat(event["start_date"].strip())
            if any(
                abs((anomaly_start - event_date).days) <= arguments.proximity_days
                for anomaly_start in anomaly_starts
            ):
                labels.append(
                    normalize_event(event, "country", "community", "automatic")
                )
    return labels


def report_same_day_events(labels: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        groups[
            (
                label["scope"],
                label.get("country_code", ""),
                label["date"],
            )
        ].append(label)
    duplicate_groups = [
        (key, group) for key, group in groups.items() if len(group) > 1
    ]
    if duplicate_groups:
        print(
            f"warning: retained {len(duplicate_groups)} same-scope, same-country, "
            "same-day event groups for review",
            file=sys.stderr,
        )
        for (scope, country_code, start_date), group in duplicate_groups:
            descriptions = "; ".join(
                label.get("description", "(no description)") for label in group
            )
            print(
                f"  {scope} {country_code or 'all'} {start_date}: {descriptions}",
                file=sys.stderr,
            )


def semantic_deduplicate(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for label in labels:
        key = (
            label["scope"],
            label.get("country_code", "").upper(),
            label["date"],
            label.get("description", ""),
        )
        unique.setdefault(key, label)
    return sorted(
        unique.values(),
        key=lambda label: (
            label["scope"] == "global",
            label.get("country_code", ""),
            label["date"],
            label.get("description", ""),
        ),
    )


def apply_exclusions(
    labels: list[dict[str, Any]], exclusions_path: Path | None
) -> tuple[list[dict[str, Any]], int]:
    if exclusions_path is None:
        return labels, 0
    with exclusions_path.open(newline="", encoding="utf-8-sig") as input_file:
        exclusions = list(csv.DictReader(input_file))
    required = {"country_code", "start_date", "description", "category"}
    missing = required.difference(exclusions[0] if exclusions else ())
    if missing:
        raise ValueError(
            f"{exclusions_path} is missing columns: {', '.join(sorted(missing))}"
        )
    keys = {
        (
            row["country_code"].strip().upper(),
            row["start_date"].strip(),
            row["description"].strip(),
            row["category"].strip(),
        )
        for row in exclusions
    }
    available = {
        (
            label.get("country_code", "").upper(),
            label["date"],
            label.get("description", ""),
            label.get("metadata", {}).get("category", ""),
        )
        for label in labels
    }
    unmatched = keys.difference(available)
    if unmatched:
        raise ValueError(
            f"{exclusions_path} contains {len(unmatched)} exclusion(s) that do not "
            "match a generated label"
        )
    retained = [
        label
        for label in labels
        if (
            label.get("country_code", "").upper(),
            label["date"],
            label.get("description", ""),
            label.get("metadata", {}).get("category", ""),
        )
        not in keys
    ]
    return retained, len(labels) - len(retained)


def main() -> None:
    arguments = parse_arguments()
    labels = load_manual_labels(arguments)
    manual_count = len(labels)
    if arguments.community_events:
        labels.extend(derive_automatic_labels(arguments))
    labels, cross_country_count = apply_cross_country_matches(
        labels, arguments.cross_country_matches
    )
    labels, excluded_count = apply_exclusions(labels, arguments.exclusions)
    labels = semantic_deduplicate(labels)
    report_same_day_events(labels)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    country_count = sum(label["scope"] == "country" for label in labels)
    global_count = len(labels) - country_count
    print(
        f"wrote {len(labels)} labels ({country_count} country, {global_count} global); "
        f"loaded {manual_count} manually verified rows; "
        f"added {cross_country_count} cross-country rules; "
        f"excluded {excluded_count} rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
