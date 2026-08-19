#!/usr/bin/env python3
"""Compare Rust detector JSON with output from `python_reference.py`."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from python_reference import run_reference


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument(
        "--algorithm",
        choices=("chebyshev", "isolation-forest", "lof", "ensemble"),
        default="chebyshev",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def close(left, right, relative_tolerance: float, absolute_tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        left,
        right,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def main() -> None:
    arguments = parse_arguments()
    completed = subprocess.run(
        [
            arguments.rust_binary,
            "detect",
            "--input",
            arguments.input,
            "--parameters",
            arguments.parameters,
            "--algorithm",
            arguments.algorithm,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rust = json.loads(completed.stdout)["anomalies"]
    python = run_reference(arguments.input, arguments.parameters, arguments.algorithm)

    errors: list[str] = []
    if len(rust) != len(python):
        errors.append(f"anomaly count differs: Rust={len(rust)}, Python={len(python)}")

    for index, (rust_anomaly, python_anomaly) in enumerate(zip(rust, python)):
        for field in ("start", "end", "peak"):
            if rust_anomaly[field] != python_anomaly[field]:
                errors.append(
                    f"anomaly {index} {field}: "
                    f"Rust={rust_anomaly[field]}, Python={python_anomaly[field]}"
                )
        for field in ("score", "residual", "impact"):
            if not close(
                rust_anomaly[field],
                python_anomaly[field],
                arguments.relative_tolerance,
                arguments.absolute_tolerance,
            ):
                errors.append(
                    f"anomaly {index} {field}: "
                    f"Rust={rust_anomaly[field]}, Python={python_anomaly[field]}"
                )

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"matched {len(rust)} anomalies")


if __name__ == "__main__":
    main()
