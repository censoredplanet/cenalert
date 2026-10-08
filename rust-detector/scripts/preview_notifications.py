#!/usr/bin/env python3
"""Render notification previews from a snapshot without contacting Slack."""

import argparse
import json
import math
import os
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def metric_text(anomaly, algorithm):
    if anomaly.get("sparse"):
        return f"Croston residual: {anomaly['residual']:.2f}"
    if algorithm == "ensemble":
        votes = anomaly.get("ensemble_vote")
        if votes:
            names = ("chebyshev", "isolation_forest", "lof")
            agreeing = [name for name in names if votes[name]["voted"]]
            return f"Ensemble votes at onset: {votes['vote_count']}/{len(names)} ({', '.join(agreeing)})"
        return f"Ensemble votes at onset: {anomaly['score']:g}/3"
    return f"{algorithm} score at onset: {anomaly['score']:.2f}"


def event_status(anomaly, as_of, last_date):
    yesterday = as_of - timedelta(days=1)
    if last_date < yesterday:
        return None
    start = date.fromisoformat(anomaly["start"])
    end = date.fromisoformat(anomaly["end"])
    if start == yesterday:
        return "started"
    if start < yesterday and end >= yesterday:
        return "ongoing"
    if end == yesterday - timedelta(days=1):
        return "ended"
    return None


def plot_event(connection, country, anomaly, parameters, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = date.fromisoformat(anomaly["start"])
    end = date.fromisoformat(anomaly["end"])
    beginning = start - timedelta(days=int(parameters["window"]))
    points = [json.loads(row[0]) for row in connection.execute(
        "SELECT payload FROM annotated WHERE country=? AND date BETWEEN ? AND ? ORDER BY date",
        (country, beginning.isoformat(), end.isoformat()))]
    if not points:
        raise ValueError(f"no annotated data for {country} {start}")
    dates = [date.fromisoformat(point["date"]) for point in points]
    values = [point["value"] for point in points]
    thresholds = [point.get("threshold") if point.get("threshold") is not None else math.nan
                  for point in points]
    fig, ax = plt.subplots(figsize=(8, 3))
    try:
        ax.plot(dates, values, color="black", label="Search volume")
        if any(math.isfinite(value) for value in thresholds):
            ax.plot(dates, thresholds, "--", marker=".", color="darkgreen", label="Threshold")
        if start == end:
            ax.axvline(start, color="red", alpha=0.3)
        else:
            ax.axvspan(start, end, color="red", alpha=0.12)
        ax.set(title=f"{country}: {start} to {end}", ylabel="Search volume")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def preview(database, output, as_of, plots=False):
    if output.exists():
        raise ValueError("preview directory already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("schema_version") != "2":
            raise ValueError("notification preview requires snapshot schema version 2")
        messages = []
        stale = []
        with tempfile.TemporaryDirectory(prefix=".notification-preview-", dir=output.parent) as work:
            stage = Path(work) / "preview"
            stage.mkdir()
            for country_row in connection.execute("SELECT * FROM countries ORDER BY country"):
                country = country_row["country"]
                if len(country) != 2 or not country.isascii() or not country.isalpha() or country != country.upper():
                    raise ValueError("invalid country code in snapshot")
                last = connection.execute("SELECT max(date) FROM series WHERE country=?", (country,)).fetchone()[0]
                if last is None or date.fromisoformat(last) < as_of - timedelta(days=1):
                    stale.append(country)
                    continue
                for row in connection.execute("SELECT * FROM events WHERE country=? ORDER BY start", (country,)):
                    anomaly = json.loads(row["payload"])
                    status = event_status(anomaly, as_of, date.fromisoformat(last))
                    if not status:
                        continue
                    name = f"{country}_{anomaly['start']}_{status}"
                    text = [f"VPN search spike {status} in {country}",
                            f"Topic: {country_row['topic']}",
                            f"Period: {anomaly['start']} to {anomaly['end']}",
                            f"Peak: {anomaly['peak']}",
                            metric_text(anomaly, country_row["algorithm"]),
                            f"Impact: {anomaly['impact']:.2f}"]
                    if row["title"] or row["description"]:
                        text.append("Nearby known event: " + (row["title"] or row["description"]))
                    (stage / f"{name}.txt").write_text("\n".join(text) + "\n")
                    if plots:
                        plot_event(connection, country, anomaly, json.loads(country_row["parameters"]), stage / f"{name}.png")
                    messages.append({"country": country, "start": anomaly["start"],
                                     "status": status, "text": f"{name}.txt",
                                     "image": f"{name}.png" if plots else None})
            (stage / "manifest.json").write_text(json.dumps({
                "as_of": as_of.isoformat(), "event_layer": metadata.get("event_layer"),
                "messages": messages, "stale_countries": stale,
            }, indent=2) + "\n")
            os.replace(stage, output)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date())
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    preview(args.database, args.output_directory, args.as_of, args.plots)
    print(f"Wrote preview to {args.output_directory}; no notifications sent")


if __name__ == "__main__":
    main()
