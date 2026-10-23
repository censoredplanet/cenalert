"""Combine completed replacement runs and refresh label metrics without tuning."""
import argparse
import filecmp
import json
import shutil
from pathlib import Path

from run_experiment_matrix import evaluate_period, export_results, write_json


def read(path):
    return json.loads(path.read_text())


def integrate(base, replacements, labels_path, output):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    manifest = read(base / 'experiment.json')
    labels = read(labels_path)
    overrides = {}
    for source in replacements:
        config = read(source / 'experiment.json')
        for key in manifest.keys() - {'countries', 'variants'}:
            if manifest[key] != config[key]:
                raise ValueError(f'Incompatible {key}: {source}')
        if read(source / 'inputs/labels.json') != labels:
            raise ValueError(f'Label mismatch: {source}')
        if not filecmp.cmp(base / 'inputs/cenalert-detector',
                           source / 'inputs/cenalert-detector', shallow=False):
            raise ValueError(f'Binary mismatch: {source}')
        for country in config['countries']:
            for suffix in ('.json', '-train.json', '-evaluation.json'):
                if read(base / 'inputs' / (country + suffix)) != read(source / 'inputs' / (country + suffix)):
                    raise ValueError(f'Series mismatch: {country}{suffix}')
        for path in sorted((source / 'runs').glob('*/*/*/result.json')):
            run = read(path)
            if run['status'] != 'completed':
                raise ValueError(f'Replacement incomplete: {path}')
            relative = path.parent.relative_to(source)
            if relative in overrides:
                raise ValueError(f'Duplicate replacement: {relative}')
            if read(base / relative / 'result.json')['status'] != 'no_feasible_solution':
                raise ValueError(f'Replacement was not a failed run: {relative}')
            overrides[relative] = path.parent
    shutil.copytree(base, output)
    for relative, source in overrides.items():
        shutil.copytree(source, output / relative, dirs_exist_ok=True)
    write_json(output / 'inputs/labels.json', labels)
    write_json(output / 'inputs/training-labels.json',
               [l for l in labels if l['date'] <= manifest['train_end']])
    runs = []
    for path in sorted((output / 'runs').glob('*/*/*/result.json')):
        run = read(path)
        if run['status'] == 'completed':
            for selection in run['selections']:
                anomalies = read(path.parent / f"{selection['selection']}-detection.json")['anomalies']
                selection.update(evaluate_period(anomalies, labels, run['country'],
                    run['test_first_date'], run['test_last_date'], manifest['proximity_days']))
                for year in selection['yearly']:
                    year.update(evaluate_period(anomalies, labels, run['country'],
                        year['first_date'], year['last_date'], manifest['proximity_days']))
            write_json(path, run)
        elif not run['status'].startswith('skipped_'):
            raise ValueError(f'Unresolved run: {path}')
        runs.append(run)
    export_results(output, runs, manifest['proximity_days'])
    write_json(output / 'integration.json', {
        'base': str(base), 'evaluation_labels': str(labels_path),
        'replacements': {str(k): str(v) for k, v in overrides.items()},
        'base_tuning_labels': read(base / 'inputs/training-labels.json'),
        'replacement_tuning_labels': [l for l in labels if l['date'] <= manifest['train_end']],
    })
    print(f'Integrated {len(overrides)} replacement runs; {len(runs)} configurations total.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--replacements', type=Path, nargs='+', required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    integrate(args.base.resolve(), [p.resolve() for p in args.replacements],
              args.labels.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
