#!/usr/bin/env python3
"""Download current All Time CenAlert series from the public Dashboard API."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as calendar_date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "https://data.censoredplanet.org/query"
COUNTRIES_QUERY = """
query GetCenalertCountries {
  cenalertCountries
}
"""
TIMESERIES_QUERY = """
query Timeseries($country: String!) {
  cenalertTimeseries(country: $country, range: null) {
    country
    rate: value
    date
  }
}
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download All Time vpn series from the CenAlert Dashboard."
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Optional ISO alpha-2 country-code subset (default: every country).",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if arguments.retries < 0:
        parser.error("--retries must be non-negative")
    return arguments


def graphql_request(
    endpoint: str,
    query: str,
    variables: dict[str, Any] | None,
    retries: int,
) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cenalert-dashboard-series-downloader/1.0",
        },
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if result.get("errors"):
                messages = "; ".join(
                    str(error.get("message", error)) for error in result["errors"]
                )
                raise RuntimeError(f"GraphQL error: {messages}")
            data = result.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("GraphQL response does not contain a data object")
            return data
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == retries:
                raise RuntimeError(f"request failed after {retries + 1} attempts") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def available_countries(endpoint: str, retries: int) -> list[str]:
    data = graphql_request(endpoint, COUNTRIES_QUERY, None, retries)
    values = data.get("cenalertCountries")
    if not isinstance(values, list):
        raise RuntimeError("cenalertCountries is not a list")
    countries = sorted({str(value).strip().upper() for value in values if value})
    invalid = [country for country in countries if len(country) != 2]
    if invalid:
        raise RuntimeError(f"invalid country codes returned by API: {invalid}")
    return countries


def download_country(
    endpoint: str, country: str, destination: Path, retries: int
) -> dict[str, Any]:
    data = graphql_request(
        endpoint, TIMESERIES_QUERY, {"country": country}, retries
    )
    records = data.get("cenalertTimeseries")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"{country}: API returned no observations")

    observations = []
    seen_dates = set()
    for record in records:
        date = str(record["date"])[:10]
        try:
            calendar_date.fromisoformat(date)
        except ValueError as error:
            raise RuntimeError(f"{country}: invalid observation date {date}") from error
        if date in seen_dates:
            raise RuntimeError(f"{country}: duplicate observation for {date}")
        seen_dates.add(date)
        record_country = str(record.get("country", country)).strip().upper()
        if record_country != country:
            raise RuntimeError(
                f"{country}: response contains observation for {record_country}"
            )
        rate = float(record["rate"])
        if not math.isfinite(rate) or rate < 0:
            raise RuntimeError(f"{country}: invalid rate {rate} on {date}")
        observations.append(
            {
                "country": country,
                "date": date,
                "rate": rate,
                "topic": "vpn",
            }
        )
    observations.sort(key=lambda observation: observation["date"])

    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{country}.json"
    temporary = destination / f".{country}.json.tmp"
    temporary.write_text(json.dumps(observations, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output)
    return {
        "country": country,
        "observations": len(observations),
        "first_date": observations[0]["date"],
        "last_date": observations[-1]["date"],
    }


def main() -> None:
    arguments = parse_arguments()
    supported = available_countries(arguments.endpoint, arguments.retries)
    requested = {
        country.strip().upper() for country in (arguments.countries or supported)
    }
    unknown = sorted(requested.difference(supported))
    if unknown:
        raise SystemExit(f"countries not offered by the Dashboard: {', '.join(unknown)}")

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                download_country,
                arguments.endpoint,
                country,
                arguments.output_directory,
                arguments.retries,
            ): country
            for country in sorted(requested)
        }
        for future in as_completed(futures):
            country = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"{country}: {result['observations']} observations "
                    f"({result['first_date']} through {result['last_date']})"
                )
            except Exception as error:  # Keep other country downloads running.
                failures.append((country, str(error)))
                print(f"{country}: ERROR: {error}")

    summary = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": arguments.endpoint,
        "topic": "vpn",
        "range": "all",
        "countries": sorted(results, key=lambda result: result["country"]),
        "failures": [
            {"country": country, "error": error} for country, error in failures
        ],
    }
    summary_path = arguments.output_directory.parent / (
        arguments.output_directory.name + "-manifest.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(results)} of {len(requested)} countries")
    print(f"Manifest: {summary_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
