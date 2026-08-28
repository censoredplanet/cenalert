#!/usr/bin/env python3
"""Compare Rust and Python detectors across a directory of paper series."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from python_reference import run_reference


FIXED_PARAMETERS: dict[str, Any] = {
    "isolation-forest": [60, 0.8, 5, 0.05],
    "lof": [60, 1.5, 5, 0.05],
    "ensemble": {
        "window": 60,
        "min_residual_ratio": 0.05,
        "efficiency": 0.05,
        "chebyshev_z": 4,
        "chebyshev_k": 6,
        "isolation_forest_min_score": 0.8,
        "lof_min_score": 1.5,
    },
}


@dataclass
class Result:
    country: str
    algorithm: str
    observations: int
    zero_observations: int
    python_anomalies: int | None
    rust_anomalies: int | None
    dates_match: bool | None
    metadata_match: bool | None
    max_metadata_difference: float | None
    status: str
    detail: str = ""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-directory", required=True, type=Path)
    parser.add_argument("--chebyshev-parameters", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("chebyshev", "isolation-forest", "lof", "ensemble"),
        default=("chebyshev", "isolation-forest", "lof", "ensemble"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--outlier-impact-tolerance", type=float, default=0.03)
    return parser.parse_args()


def convert_series(source: Path, destination: Path, country: str) -> tuple[int, int]:
    records = []
    zero_observations = 0
    with source.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            value = float(row["value"])
            zero_observations += math.isclose(value, 0.0, abs_tol=1e-12)
            records.append(
                {
                    "date": row["date"],
                    "value": value,
                    "country": country,
                    "topic": "012t0g",
                }
            )
    destination.write_text(json.dumps(records))
    return len(records), zero_observations


def close(left: Any, right: Any, relative: float, absolute: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=relative, abs_tol=absolute)


def run_one(
    country: str,
    algorithm: str,
    input_path: str,
    parameter_path: str,
    rust_binary: str,
    observations: int,
    zero_observations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Result:
    try:
        python = run_reference(Path(input_path), Path(parameter_path), algorithm)
        completed = subprocess.run(
            [
                rust_binary,
                "detect",
                "--algorithm",
                algorithm,
                "--input",
                input_path,
                "--parameters",
                parameter_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rust = json.loads(completed.stdout)["anomalies"]
    except Exception as error:  # Keep the remainder of the country matrix running.
        return Result(
            country,
            algorithm,
            observations,
            zero_observations,
            None,
            None,
            None,
            None,
            None,
            "error",
            f"{type(error).__name__}: {error}",
        )

    dates_match = len(python) == len(rust)
    metadata_match = dates_match
    maximum_difference = 0.0
    details = []
    for index, (python_anomaly, rust_anomaly) in enumerate(zip(python, rust)):
        for field in ("start", "end", "peak"):
            if python_anomaly[field] != rust_anomaly[field]:
                dates_match = False
                details.append(f"anomaly {index} {field}")
        for field in ("score", "residual", "impact"):
            left = python_anomaly[field]
            right = rust_anomaly[field]
            if left is not None and right is not None:
                maximum_difference = max(maximum_difference, abs(left - right))
            if not close(left, right, relative_tolerance, absolute_tolerance):
                metadata_match = False
                details.append(f"anomaly {index} {field}")

    if len(python) != len(rust):
        details.insert(0, f"count {len(python)} != {len(rust)}")
    status = "match" if dates_match and metadata_match else "mismatch"
    return Result(
        country,
        algorithm,
        observations,
        zero_observations,
        len(python),
        len(rust),
        dates_match,
        metadata_match,
        maximum_difference,
        status,
        "; ".join(details[:20]),
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.workers < 1:
        raise SystemExit("--workers must be positive")
    series = sorted(arguments.series_directory.glob("*.csv"))
    if not series:
        raise SystemExit("no CSV series found")

    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="cenalert-parity-") as temporary:
        temporary_path = Path(temporary)
        input_details = {}
        for source in series:
            country = source.stem.upper()
            input_path = temporary_path / f"{country}.json"
            counts = convert_series(source, input_path, country)
            input_details[country] = (input_path, *counts)

        fixed_parameter_paths = {}
        for algorithm, parameters in FIXED_PARAMETERS.items():
            path = temporary_path / f"{algorithm}-parameters.json"
            path.write_text(json.dumps(parameters))
            fixed_parameter_paths[algorithm] = path

        jobs = []
        for country, (input_path, observations, zeros) in input_details.items():
            for algorithm in arguments.algorithms:
                if algorithm == "ensemble" and observations == zeros:
                    results.append(
                        Result(
                            country,
                            algorithm,
                            observations,
                            zeros,
                            None,
                            None,
                            None,
                            None,
                            None,
                            "skipped",
                            "all-zero series",
                        )
                    )
                    continue
                parameter_path = (
                    arguments.chebyshev_parameters / f"{country}.json"
                    if algorithm == "chebyshev"
                    else fixed_parameter_paths[algorithm]
                )
                if not parameter_path.exists():
                    results.append(
                        Result(
                            country,
                            algorithm,
                            observations,
                            zeros,
                            None,
                            None,
                            None,
                            None,
                            None,
                            "skipped",
                            "parameter file not available",
                        )
                    )
                    continue
                absolute = (
                    arguments.outlier_impact_tolerance
                    if algorithm in ("isolation-forest", "ensemble")
                    else arguments.absolute_tolerance
                )
                jobs.append(
                    (
                        country,
                        algorithm,
                        str(input_path),
                        str(parameter_path),
                        str(arguments.rust_binary.resolve()),
                        observations,
                        zeros,
                        arguments.relative_tolerance,
                        absolute,
                    )
                )

        with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
            futures = [executor.submit(run_one, *job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{completed}/{len(futures)}] "
                    f"{result.country} {result.algorithm}: {result.status}",
                    flush=True,
                )

    results.sort(key=lambda result: (result.country, result.algorithm))
    summary = {
        status: sum(result.status == status for result in results)
        for status in ("match", "mismatch", "error", "skipped")
    }
    payload = {
        "series": len(series),
        "algorithms": list(arguments.algorithms),
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if summary["mismatch"] or summary["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
