#!/usr/bin/env python3
"""Measure steady-state Python and release-mode Rust detector runtimes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from python_reference import run_reference


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def time_call(function, iterations: int) -> list[float]:
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    return durations


def main() -> None:
    arguments = parse_arguments()
    rust_command = [
        arguments.rust_binary,
        "detect",
        "--input",
        arguments.input,
        "--parameters",
        arguments.parameters,
    ]

    def run_rust() -> None:
        subprocess.run(rust_command, check=True, capture_output=True)

    def run_python() -> None:
        run_reference(arguments.input, arguments.parameters)

    # Exclude one-time dynamic linking, filesystem cache, and Numba compilation.
    run_rust()
    run_python()

    rust_durations = time_call(run_rust, arguments.iterations)
    python_durations = time_call(run_python, arguments.iterations)
    rust_median = statistics.median(rust_durations)
    python_median = statistics.median(python_durations)

    print(
        json.dumps(
            {
                "iterations": arguments.iterations,
                "rust_seconds": rust_durations,
                "python_seconds": python_durations,
                "rust_median_seconds": rust_median,
                "python_median_seconds": python_median,
                "median_speedup": python_median / rust_median,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
