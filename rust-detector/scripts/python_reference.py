#!/usr/bin/env python3
"""Run CenAlert's current Python Chebyshev detector on Dashboard JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from lib.detection import ChebyshevInequality  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    return parser.parse_args()


def parse_date(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def json_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def run_reference(input_path: Path, parameters_path: Path) -> list[dict[str, Any]]:
    records = json.loads(input_path.read_text())
    parameters = json.loads(parameters_path.read_text())
    parameters[0] = round(parameters[0])

    series = pl.DataFrame(
        {
            "date": [parse_date(record["date"]) for record in records],
            "value": [
                float(record.get("rate", record.get("value"))) for record in records
            ],
        }
    ).sort("date")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector = ChebyshevInequality(*parameters)
        detector.run(series)
        anomalies = detector.anomalies()

    return [
        {
            "start": row[0].isoformat(),
            "end": row[1].isoformat(),
            "peak": row[2].isoformat(),
            "score": json_number(row[3]),
            "residual": json_number(row[4]),
            "impact": json_number(row[5]),
        }
        for row in anomalies.iter_rows()
    ]


def main() -> None:
    arguments = parse_arguments()
    output = run_reference(arguments.input, arguments.parameters)
    json.dump({"anomalies": output}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
