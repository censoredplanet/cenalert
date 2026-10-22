"""Prepare raw CSV windows for the Rust rebuild-series command."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path


def read_samples(paths):
    samples = []
    for path in paths:
        with path.open(newline="") as stream:
            samples.append([
                {"date": date.fromisoformat(row["date"]).isoformat(), "value": float(row["value"])}
                for row in csv.DictReader(stream)
            ])
    return samples


def prepare(raw_directory, country, as_of, sample_count=45, topic="vpn"):
    if sample_count < 1 or len(country) != 2 or not country.isalpha():
        raise ValueError("expected a two-letter country code and positive sample count")
    latest = {}
    for batch in sorted(raw_directory.glob("output_*")):
        collected = date.fromisoformat(batch.name.removeprefix("output_"))
        if collected > as_of:
            continue
        for first in sorted((batch / "sample0" / country).glob("*_multiTimeline.csv")):
            paths = [batch / f"sample{i}" / country / first.name for i in range(sample_count)]
            if all(path.is_file() for path in paths):
                latest[first.name] = (collected, paths)
    if not latest:
        raise ValueError(f"no complete sample windows for {country}")
    windows = []
    for collected, paths in sorted(latest.values(), key=lambda item: (item[0], item[1][0].name)):
        window = {
            "source": f"output_{collected}/{country}/{paths[0].name}",
            "collected_on": collected.isoformat(), "samples": read_samples(paths),
        }
        coarse_paths = [path.with_name(path.name.replace("_multiTimeline.csv", "_coarseMultiTimeline.csv")) for path in paths]
        if all(path.is_file() for path in coarse_paths):
            window["coarse"] = {
                "source": f"output_{collected}/{country}/{coarse_paths[0].name}",
                "samples": read_samples(coarse_paths),
            }
        windows.append(window)
    return {"country": country, "topic": topic, "windows": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--sample-count", type=int, default=45)
    parser.add_argument("--topic", default="vpn")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.raw_directory, args.country.upper(), args.as_of, args.sample_count, args.topic)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write("\n")
    print(f"{result['country']}: {len(result['windows'])} complete windows")


if __name__ == "__main__":
    main()
