#!/usr/bin/env python3
"""Build a dashboard snapshot from saved ensemble parameters and verify replay."""

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from contextlib import closing

from build_snapshot import build_snapshot


def publish(results, output, seed=42):
    results = results.resolve()
    experiment = json.loads((results / 'experiment.json').read_text())
    runs = results / 'runs' / str(seed)
    completed = []
    for country in experiment['countries']:
        run = runs / country / 'labels_ensemble'
        status = json.loads((run / 'result.json').read_text())['status']
        if status == 'completed':
            completed.append(country)
        elif status not in ('skipped_all_zero', 'skipped_zero_training'):
            raise ValueError(f'{country}: cannot publish incomplete run ({status})')
    if not completed:
        raise ValueError('No completed countries')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.replay-', dir=output.parent) as temporary:
        root = Path(temporary)
        parameters = root / 'parameters'
        series = root / 'series'
        parameters.mkdir()
        series.mkdir()
        for country in completed:
            (parameters / f'{country}.json').symlink_to(runs / country / 'labels_ensemble/knee-parameters.json')
            (series / f'{country}.json').symlink_to(results / 'inputs' / f'{country}-evaluation.json')
        database = root / 'cenalert.sqlite'
        build_snapshot(argparse.Namespace(
            series_directory=series, parameters_directory=parameters,
            labels=results / 'inputs/labels.json', rust_binary=results / 'inputs/cenalert-detector',
            output=database, countries=completed, topic='vpn', algorithm='ensemble',
            proximity_days=experiment.get('proximity_days', 3), reference_end=experiment['train_end']))
        total = 0
        with closing(sqlite3.connect(database)) as connection, connection:
            for country in completed:
                saved = json.loads((runs / country / 'labels_ensemble/knee-detection.json').read_text())
                replayed = [json.loads(row[0]) for row in connection.execute(
                    'SELECT payload FROM events WHERE country=? ORDER BY start', (country,))]
                expected = sorted(saved['anomalies'], key=lambda event: event['start'])
                if len(replayed) != len(expected):
                    raise ValueError(f'{country}: replay event count differs from saved result')
                for actual, previous in zip(replayed, expected):
                    for field in ('start', 'end', 'peak'):
                        if actual[field] != previous[field]:
                            raise ValueError(f'{country}: replay differs on {field}')
                    for field in ('impact', 'score', 'residual'):
                        if not math.isclose(actual[field], previous[field], rel_tol=1e-9, abs_tol=1e-10):
                            raise ValueError(f'{country}: replay differs on {field}')
                total += len(replayed)
            connection.executemany('INSERT INTO metadata VALUES (?, ?)', [
                ('source_results', str(results)), ('selection', 'knee'),
                ('variant', 'labels_ensemble'), ('seed', str(seed)),
                ('replay_verified_countries', str(len(completed)))])
        with database.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(database, output)
    print(f'Published {len(completed)} countries and {total} verified events: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    publish(args.results, args.output, args.seed)
