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

from lib.detection import (  # noqa: E402
    AnomalyDetector,
    ChebyshevInequality,
    IsolationForest,
    LocalOutlierFactor,
)


class Ensemble(AnomalyDetector):
    """Three-detector majority-vote reference used for parity validation."""

    def __init__(self, parameters: dict[str, float], series_maximum: float):
        window = int(parameters["window"])
        min_residual = (
            float(parameters["min_residual_ratio"]) * series_maximum
            if "min_residual_ratio" in parameters
            else float(parameters["min_residual"])
        )
        efficiency = float(parameters["efficiency"])
        super().__init__(window, min_residual=min_residual, efficiency=efficiency)
        self._min_score = 2
        self.detectors = [
            ChebyshevInequality(
                window,
                float(parameters["chebyshev_z"]),
                float(parameters["chebyshev_k"]),
                min_residual,
                efficiency,
            ),
            IsolationForest(
                window,
                float(parameters["isolation_forest_min_score"]),
                min_residual,
                efficiency,
            ),
            LocalOutlierFactor(
                window,
                float(parameters["lof_min_score"]),
                min_residual,
                efficiency,
            ),
        ]

    def score(self, value):
        for detector in self.detectors:
            detector._window = self._window
        scores = [detector.score(value) for detector in self.detectors]
        return sum(
            score >= detector._min_score
            for score, detector in zip(scores, self.detectors)
        )

    def threshold(self, initial_guess):
        return min(detector.threshold(initial_guess) for detector in self.detectors)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument(
        "--algorithm",
        choices=("chebyshev", "isolation-forest", "lof", "ensemble"),
        default="chebyshev",
    )
    return parser.parse_args()


def parse_date(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def json_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def run_reference(
    input_path: Path, parameters_path: Path, algorithm: str = "chebyshev"
) -> list[dict[str, Any]]:
    records = json.loads(input_path.read_text())
    parameters = json.loads(parameters_path.read_text())
    if algorithm != "ensemble" and isinstance(parameters, list):
        parameters[0] = round(parameters[0])

    series = pl.DataFrame(
        {
            "date": [parse_date(record["date"]) for record in records],
            "value": [
                float(record.get("rate", record.get("value"))) for record in records
            ],
        }
    ).sort("date")
    series_maximum = float(series["value"].max())

    if isinstance(parameters, dict) and algorithm != "ensemble":
        min_residual = (
            float(parameters["min_residual_ratio"]) * series_maximum
            if "min_residual_ratio" in parameters
            else float(parameters["min_residual"])
        )
        if algorithm == "chebyshev":
            parameters = [
                int(parameters["window"]),
                float(parameters["z"]),
                float(parameters["k"]),
                min_residual,
                float(parameters["efficiency"]),
            ]
        else:
            parameters = [
                int(parameters["window"]),
                float(parameters["min_score"]),
                min_residual,
                float(parameters["efficiency"]),
            ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if algorithm == "chebyshev":
            detector = ChebyshevInequality(*parameters)
        elif algorithm == "isolation-forest":
            detector = IsolationForest(*parameters)
        elif algorithm == "lof":
            detector = LocalOutlierFactor(*parameters)
        else:
            detector = Ensemble(parameters, series_maximum)
        if algorithm != "chebyshev":
            # The paper implementation computes this expensive inverse-score
            # threshold only when an anomaly begins. The deployment snapshot's
            # unconditional mode passes reversed bounds to SciPy for ordinary
            # below-mean points and cannot complete an outlier-detector run.
            detector._always_compute_threshold = False
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
    output = run_reference(arguments.input, arguments.parameters, arguments.algorithm)
    json.dump({"anomalies": output}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
