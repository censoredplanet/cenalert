#!/usr/bin/env python3
"""Export a SQLite snapshot to per-country detector CSV files."""

import argparse
import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path


ANNOTATED_COLUMNS = {
    "date": "date", "value": "value", "anomaly": "anomaly", "score": "score",
    "residual": "residual", "threshold": "threshold", "min_score": "minimum_score",
    "cov2": "coefficient_of_variation_squared", "adi": "average_interdemand_interval",
    "demand_pattern": "demand_category",
}
ANOMALY_COLUMNS = ("start", "end", "peak", "score", "residual", "impact", "quartile")


def export_snapshot(database, output):
    if output.exists():
        raise ValueError("output directory already exists; choose a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if version is None or version[0] != "2":
            raise ValueError("CSV export requires snapshot schema version 2")
        with tempfile.TemporaryDirectory(prefix=".cenalert-export-", dir=output.parent) as work:
            stage = Path(work) / "output"
            stage.mkdir()
            for record in connection.execute("SELECT * FROM countries ORDER BY country"):
                country = record["country"]
                if len(country) != 2 or not country.isascii() or not country.isalpha() or country != country.upper():
                    raise ValueError("invalid country in snapshot")
                destination = stage / country
                destination.mkdir()
                (destination / "parameters.json").write_text(record["parameters"] + "\n")
                (destination / "detector.json").write_text(json.dumps({
                    "algorithm": record["algorithm"], "topic": record["topic"],
                    "scale_factor": record["scale_factor"],
                }, indent=2) + "\n")
                with (destination / "annotated.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["index", *ANNOTATED_COLUMNS])
                    writer.writeheader()
                    for index, row in enumerate(connection.execute(
                            "SELECT payload FROM annotated WHERE country=? ORDER BY date", (country,))):
                        point = json.loads(row[0])
                        values = {column: point.get(key) for column, key in ANNOTATED_COLUMNS.items()}
                        values["anomaly"] = str(bool(point["anomaly"])).lower()
                        writer.writerow({"index": index, **values})
                rows = connection.execute(
                    "SELECT * FROM events WHERE country=? ORDER BY impact, start", (country,)).fetchall()
                with (destination / "anomalies.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=ANOMALY_COLUMNS)
                    writer.writeheader()
                    for index, row in enumerate(rows):
                        values = {column: row[column] for column in ANOMALY_COLUMNS if column != "quartile"}
                        values["quartile"] = index if len(rows) < 4 else index * 4 // len(rows)
                        writer.writerow(values)
            os.replace(stage, output)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    export_snapshot(args.database, args.output_directory)
    print(f"Wrote {args.output_directory}")


if __name__ == "__main__":
    main()
