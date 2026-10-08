#!/usr/bin/env python3
"""Download/stitch, detect, and atomically publish a complete daily run."""

import argparse
import csv
import fcntl
import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from build_snapshot import build_snapshot, read_series
from export_snapshot import export_snapshot
from preview_notifications import preview

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true")
    source.add_argument("--samples-directory", type=Path)
    source.add_argument("--series-directory", type=Path)
    parser.add_argument("--previous-stitched", type=Path,
                        help="Previous normal/coarse CSVs for incremental stitching")
    parser.add_argument("--parameters-directory", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, default=Path("target/release/cenalert-detector"))
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--topic", default="vpn")
    parser.add_argument("--algorithm", choices=["ensemble", "chebyshev", "lof", "isolation-forest"], default="ensemble")
    parser.add_argument("--sample-count", type=int, default=45)
    parser.add_argument("--start-month", help="Initial collection start (YYYY-MM); default 2011-01")
    parser.add_argument("--through-date", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date() - timedelta(days=1))
    parser.add_argument("--api-key-file", type=Path,
                        help="Alternatively set GOOGLE_TRENDS_API_KEY")
    parser.add_argument("--proximity-days", type=int, default=3)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Validate local configuration without running stages")
    args = parser.parse_args(argv)
    if args.sample_count < 1 or args.proximity_days < 0:
        parser.error("sample-count must be positive; proximity-days must be non-negative")
    if args.series_directory and args.previous_stitched:
        parser.error("previous-stitched is only used with download or samples-directory")
    if args.start_month:
        try:
            month = datetime.strptime(args.start_month, "%Y-%m").date()
        except ValueError:
            parser.error("start-month must be YYYY-MM")
        if month > args.through_date:
            parser.error("start-month is after through-date")
    return args


def validate(args):
    if not args.countries or len(set(args.countries)) != len(args.countries):
        raise ValueError("provide a nonempty, unique country list")
    for country in args.countries:
        if len(country) != 2 or not country.isascii() or not country.isalpha() or country != country.upper():
            raise ValueError("countries must be uppercase two-letter codes")
        json.loads((args.parameters_directory / f"{country}.json").read_text())
        if args.previous_stitched:
            for suffix in ("_stitched.csv", "_coarse_stitched.csv"):
                if not (args.previous_stitched / f"{country}{suffix}").is_file():
                    raise ValueError(f"previous stitched file missing: {country}{suffix}")
    labels = json.loads(args.labels.read_text())
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a nonempty JSON array")
    if not args.rust_binary.is_file() or not os.access(args.rust_binary, os.X_OK):
        raise ValueError("build the release detector first or specify --rust-binary")
    for directory in (args.series_directory, args.samples_directory):
        if directory and not directory.is_dir():
            raise ValueError(f"input directory not found: {directory}")
    if args.download and not (args.api_key_file or os.environ.get("GOOGLE_TRENDS_API_KEY")):
        raise ValueError("download requires --api-key-file or GOOGLE_TRENDS_API_KEY")
    if args.api_key_file and not args.api_key_file.read_text().strip():
        raise ValueError("API key file is empty")


def download(args, output):
    from dateutil.relativedelta import relativedelta
    from googleapiclient.discovery import build
    from lib.google_trends_utils import DISCOVERY_URL, get_multi_timeline_windows

    key = (args.api_key_file.read_text() if args.api_key_file else os.environ["GOOGLE_TRENDS_API_KEY"]).strip()
    try:
        service = build("trends", "v1beta", developerKey=key,
                        discoveryServiceUrl=DISCOVERY_URL, cache_discovery=False)
    except Exception:
        raise RuntimeError("Google Trends client initialization failed; check API access") from None
    if args.previous_stitched:
        start = args.start_month or (args.through_date - relativedelta(months=10)).strftime("%Y-%m")
        window_start = max(start, (args.through_date - relativedelta(months=7)).strftime("%Y-%m"))
    else:
        start = window_start = args.start_month or "2011-01"
    try:
        for index in range(args.sample_count):
            for country in args.countries:
                destination = output / f"sample{index}" / country
                destination.mkdir(parents=True)
                get_multi_timeline_windows(service, logging.getLogger(__name__), str(destination),
                    start, window_start, args.through_date.strftime("%Y-%m"), country,
                    args.topic, window=8, overlap=7, strict=True)
    finally:
        service.close()


def stitch(args, samples, output):
    import pandas as pd
    from lib.stitching import combine_and_stitch, combine_window_pair
    from stitching_job import process_country, sorted_files

    output.mkdir()
    for country in args.countries:
        paths = [samples / f"sample{i}" / country for i in range(args.sample_count)]
        expected = None
        window_dates = {}
        for path in paths:
            normal, coarse = sorted_files(str(path), False), sorted_files(str(path), True)
            if not normal or [name.replace("_multiTimeline.csv", "") for name in normal] != [
                    name.replace("_coarseMultiTimeline.csv", "") for name in coarse]:
                raise ValueError(f"missing or unpaired sample windows for {country}")
            if expected is not None and expected != (normal, coarse):
                raise ValueError(f"inconsistent sample windows for {country}")
            expected = (normal, coarse)
            for name in normal + coarse:
                days = [row["date"] for row in read_series(path / name, country, args.topic)]
                if name in window_dates and window_dates[name] != days:
                    raise ValueError(f"inconsistent sample dates for {country}: {name}")
                window_dates[name] = days
        if args.previous_stitched:
            process_country(country, [str(path) for path in paths], str(args.previous_stitched), str(output))
        else:
            combine_and_stitch([str(path) for path in paths]).to_csv(output / f"{country}_stitched.csv", index=False)
            windows = [[pd.read_csv(path / name, parse_dates=["date"]) for name in expected[1]] for path in paths]
            merged = [combine_window_pair(windows, index, range(len(paths)))[1]
                      for index in range(len(expected[1]))]
            pd.concat(merged).drop_duplicates(subset="date").sort_values("date").to_csv(
                output / f"{country}_coarse_stitched.csv", index=False)


def normalize_series(args, source, destination):
    destination.mkdir(exist_ok=True)
    coverage = {}
    for country in args.countries:
        paths = [source / f"{country}{suffix}" for suffix in ("_stitched.csv", ".csv", ".json")]
        paths = [path for path in paths if path.is_file()]
        if len(paths) != 1:
            raise ValueError(f"expected one series for {country}")
        rows = [row for row in read_series(paths[0], country, args.topic)
                if row["date"] <= args.through_date.isoformat()]
        if not rows or rows[-1]["date"] != args.through_date.isoformat():
            raise ValueError(f"{country}: series does not reach {args.through_date}; refusing stale publication")
        dates = [date.fromisoformat(row["date"]) for row in rows]
        if any((right - left).days != 1 for left, right in zip(dates, dates[1:])):
            raise ValueError(f"{country}: series contains missing dates")
        with (destination / f"{country}_stitched.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["date", "value"])
            writer.writerows((row["date"], row["rate"]) for row in rows)
        coverage[country] = {"start": rows[0]["date"], "end": rows[-1]["date"], "observations": len(rows)}
    return coverage


@contextmanager
def staged_run(output_root):
    with tempfile.TemporaryDirectory(prefix=".daily-", dir=output_root) as work:
        stage = Path(work) / "run"
        stage.mkdir(mode=0o700)
        try:
            yield stage
        except BaseException:
            if stage.exists():
                retained = output_root / ("failed-" + Path(work).name.removeprefix(".daily-"))
                os.rename(stage, retained)
                print(f"Run failed; inputs and partial outputs retained at {retained}", file=sys.stderr)
            raise


def run(args):
    validate(args)
    if args.plan:
        print("Validated: input → stitching (if needed) → fixed-parameter detection → SQLite → CSV → local preview")
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".daily.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another daily run is using this output root") from None
        current = args.output_root / "current"
        if current.exists() and not current.is_symlink():
            raise ValueError("output-root/current must be a symlink or absent")
        with staged_run(args.output_root) as stage:
            source = args.series_directory
            if source is None:
                samples = args.samples_directory or stage / "samples"
                if args.download:
                    print("Downloading Google Trends samples", flush=True)
                    download(args, samples)
                print("Stitching samples", flush=True)
                source = stage / "stitched"
                stitch(args, samples, source)
            coverage = normalize_series(args, source, stage / "stitched")
            parameters = stage / "parameters"
            parameters.mkdir()
            for country in args.countries:
                (parameters / f"{country}.json").write_bytes((args.parameters_directory / f"{country}.json").read_bytes())
            (stage / "labels.json").write_bytes(args.labels.read_bytes())
            print("Running fixed-parameter detection", flush=True)
            build_snapshot(argparse.Namespace(series_directory=stage / "stitched",
                parameters_directory=parameters, labels=stage / "labels.json",
                rust_binary=args.rust_binary, output=stage / "cenalert.sqlite",
                countries=args.countries, topic=args.topic, algorithm=args.algorithm,
                proximity_days=args.proximity_days))
            export_snapshot(stage / "cenalert.sqlite", stage / "detection")
            preview(stage / "cenalert.sqlite", stage / "preview", args.through_date + timedelta(days=1), args.plots)
            manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), topic=args.topic,
                algorithm=args.algorithm, through_date=args.through_date.isoformat(), coverage=coverage,
                source="download" if args.download else "samples" if args.samples_directory else "series",
                sample_count=args.sample_count if not args.series_directory else None,
                previous_stitched=str(args.previous_stitched.resolve()) if args.previous_stitched else None,
                event_layer="raw", notifications_sent=False)
            (stage / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
            name = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
            destination = args.output_root / name
            os.rename(stage, destination)
            link = stage.parent / "current"
            link.symlink_to(name)
            os.replace(link, current)
            print(f"Published {destination}; current/cenalert.sqlite is ready for ingestion")


if __name__ == "__main__":
    run(arguments())
