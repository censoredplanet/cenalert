#!/usr/bin/env python3
"""Run fixed-parameter detection and publish a complete SQLite snapshot."""

import argparse
import csv
import json
import math
import os
import sqlite3
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from batch_ensemble_tuning import label_country_code, matched_label


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE labels (
    label_index INTEGER PRIMARY KEY,
    stable_id TEXT UNIQUE,
    payload TEXT NOT NULL CHECK(json_valid(payload))
);
CREATE TABLE countries (
    country TEXT PRIMARY KEY, topic TEXT NOT NULL, algorithm TEXT NOT NULL,
    parameters TEXT NOT NULL CHECK(json_valid(parameters)), scale_factor REAL NOT NULL
);
CREATE TABLE series (
    country TEXT NOT NULL REFERENCES countries(country), date TEXT NOT NULL,
    value REAL NOT NULL, PRIMARY KEY(country, date)
);
CREATE TABLE events (
    country TEXT NOT NULL REFERENCES countries(country), start TEXT NOT NULL,
    end TEXT NOT NULL, peak TEXT NOT NULL, impact REAL NOT NULL,
    score REAL NOT NULL, residual REAL NOT NULL,
    label_index INTEGER REFERENCES labels(label_index), proximity INTEGER,
    title TEXT, description TEXT, category TEXT, cause TEXT, justification TEXT,
    affected_services TEXT NOT NULL CHECK(json_valid(affected_services)),
    sources TEXT NOT NULL CHECK(json_valid(sources)),
    payload TEXT NOT NULL CHECK(json_valid(payload)),
    PRIMARY KEY(country, start), CHECK(start <= peak AND peak <= end)
);
"""


def read_series(path, country, topic):
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = json.loads(path.read_text())
    result = []
    seen = set()
    for row in rows:
        day = date.fromisoformat(row["date"][:10]).isoformat()
        value = float(row.get("rate", row.get("value")))
        if day in seen or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid or duplicate observation: {country} {day}")
        if row.get("country", country) != country or row.get("topic", topic) != topic:
            raise ValueError(f"series country/topic disagrees with configuration: {path}")
        seen.add(day)
        result.append(dict(country=country, topic=topic, date=day, rate=value))
    if not result:
        raise ValueError(f"empty series: {path}")
    return sorted(result, key=lambda row: row["date"])


def display_fields(label):
    metadata = label.get("metadata", {})
    services = metadata.get("affected_services", [])
    # Legacy descriptions can contain commas unrelated to service boundaries.
    if isinstance(services, str):
        services = [services] if services else []
    sources = metadata.get("sources")
    if sources is None:
        sources = [{"name": metadata.get("source", ""), "url": url.strip()}
                   for url in metadata.get("link", "").split(",") if url.strip()]
    if not isinstance(services, list) or not all(isinstance(s, str) for s in services):
        raise ValueError("affected_services must be a string or array of strings")
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        raise ValueError("sources must be an array of objects")
    return (label.get("title"), label.get("description"), metadata.get("category"),
            metadata.get("cause"), metadata.get("justification"),
            json.dumps(services), json.dumps(sources))


def build_snapshot(args):
    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a nonempty array")
    for label in labels:
        date.fromisoformat(label["date"])
        if label["scope"] not in ("country", "global"):
            raise ValueError("unsupported label scope")
        code = label_country_code(label)
        if code:
            label["country_code"] = code
        if label["scope"] == "country" and not code and label.get("match_mode") != "exact":
            raise ValueError("country proximity label requires country_code")
        display_fields(label)
    countries = sorted(set(args.countries))
    inputs = []
    for country in countries:
        if len(country) != 2 or not country.isascii() or not country.isalpha() or country != country.upper():
            raise ValueError("countries must be uppercase two-letter codes")
        candidates = [args.series_directory / f"{country}{suffix}"
                      for suffix in ("_stitched.csv", ".csv", ".json")]
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one series for {country}, found {len(candidates)}")
        parameter_path = args.parameters_directory / f"{country}.json"
        parameters = json.loads(parameter_path.read_text())
        inputs.append((country, read_series(candidates[0], country, args.topic), parameters))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem enables atomic publication. Nothing touches the previous DB
    # until every requested country has succeeded and integrity checks pass.
    with tempfile.TemporaryDirectory(prefix=".cenalert-", dir=args.output.parent) as work:
        work = Path(work)
        database = work / "snapshot.sqlite"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(SCHEMA)
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                ("schema_version", "1"), ("created_at", datetime.now(timezone.utc).isoformat()),
                ("event_layer", "raw"), ("proximity_days", str(args.proximity_days)),
                ("reference_policy", "full_series_maximum"),
            ])
            for index, label in enumerate(labels):
                connection.execute("INSERT INTO labels VALUES (?, ?, ?)",
                                   (index, label.get("id"), json.dumps(label)))
            for country, rows, parameters in inputs:
                source = work / "input.json"
                parameter_file = work / "parameters.json"
                source.write_text(json.dumps(rows))
                parameter_file.write_text(json.dumps(parameters))
                process = subprocess.run([str(args.rust_binary.resolve()), "detect",
                    "--input", str(source), "--parameters", str(parameter_file),
                    "--algorithm", args.algorithm], capture_output=True, text=True, check=True)
                detection = json.loads(process.stdout)
                scale = detection["scale_factor"]
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("invalid detector scale")
                connection.execute("INSERT INTO countries VALUES (?, ?, ?, ?, ?)",
                    (country, args.topic, args.algorithm, json.dumps(detection["parameters"]), scale))
                connection.executemany("INSERT INTO series VALUES (?, ?, ?)",
                    [(country, row["date"], row["rate"] / scale) for row in rows])
                for anomaly in detection["anomalies"]:
                    if not all(isinstance(anomaly[k], (int, float)) and math.isfinite(anomaly[k])
                               for k in ("impact", "score", "residual")):
                        raise ValueError("non-finite anomaly metrics")
                    match = matched_label(anomaly, labels, country, args.proximity_days)
                    label = match["label"] if match else {}
                    connection.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (country, anomaly["start"], anomaly["end"], anomaly["peak"],
                         anomaly["impact"], anomaly["score"], anomaly["residual"],
                         labels.index(label) if match else None,
                         match["distance_days"] if match else None,
                         *display_fields(label), json.dumps(anomaly)))
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("snapshot integrity check failed")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("snapshot foreign key check failed")
        finally:
            connection.close()
        with database.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(database, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("series-directory", "parameters-directory", "labels", "rust-binary", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--topic", default="vpn")
    parser.add_argument("--algorithm", choices=["chebyshev", "ensemble", "isolation-forest", "lof"], default="ensemble")
    parser.add_argument("--proximity-days", type=int, default=3)
    args = parser.parse_args()
    if args.proximity_days < 0:
        parser.error("proximity-days must be non-negative")
    build_snapshot(args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
