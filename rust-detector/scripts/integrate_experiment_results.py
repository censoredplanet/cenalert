"""Combine completed replacement runs and refresh label metrics without tuning."""
import argparse
import filecmp
import json
import shutil
from pathlib import Path

from run_experiment_matrix import evaluate_period, export_results, write_json


def read(path):
    return json.loads(path.read_text())


def import_single_detector_runs(target, source):
    """Reuse unchanged single-detector variants after ensemble-only retuning."""
    config, original = read(target / 'experiment.json'), read(source / 'experiment.json')
    variants = ('original', 'labels', 'fixed_3sigma')
    if any(v in config['variants'] for v in variants):
        raise ValueError('Target already contains single-detector variants')
    for key in (config.keys() | original.keys()) - {'variants', 'ensemble_member_balance'}:
        if config.get(key) != original.get(key):
            raise ValueError(f'Incompatible {key}')
    for name in ('labels.json', 'training-labels.json'):
        if read(target / 'inputs' / name) != read(source / 'inputs' / name):
            raise ValueError(f'Label mismatch: {name}')
    paths = []
    for country in config['countries']:
        for suffix in ('.json', '-train.json', '-evaluation.json'):
            name = country + suffix
            if read(target / 'inputs' / name) != read(source / 'inputs' / name):
                raise ValueError(f'Series mismatch: {name}')
        for seed in config['seeds']:
            for variant in variants:
                path = Path(f'runs/{seed}/{country}/{variant}')
                run = read(source / path / 'result.json')
                if run['status'] != 'completed' and not run['status'].startswith('skipped_'):
                    raise ValueError(f'Incomplete source: {path}')
                if (target / path).exists():
                    raise ValueError(f'Target run exists: {path}')
                paths.append(path)
    for path in paths:
        shutil.copytree(source / path, target / path)
    # Keep the original executable with the imported runs' provenance.
    shutil.copy2(source / 'inputs/cenalert-detector', target / 'inputs/single-detector-binary')
    config['variants'].update({v: original['variants'][v] for v in variants})
    write_json(target / 'experiment.json', config)
    write_json(target / 'integration.json', {
        'imported_from': str(source), 'imported_variants': list(variants),
        'imported_binary': 'inputs/single-detector-binary',
        'source_experiment': original,
        'ensemble_binary': 'inputs/cenalert-detector',
    })
    runs = [read(p) for p in sorted((target / 'runs').glob('*/*/*/result.json'))]
    export_results(target, runs, config['proximity_days'])
    print(f'Imported {len(paths)} unchanged single-detector runs.')


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
    parser.add_argument('--replacements', type=Path, nargs='+')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--import-single-from', type=Path,
                        help='Import unchanged single-detector runs into --base in place')
    args = parser.parse_args()
    if args.import_single_from:
        if args.replacements or args.labels or args.output:
            parser.error('--import-single-from only accepts --base')
        import_single_detector_runs(args.base.resolve(), args.import_single_from.resolve())
        return
    if not (args.replacements and args.labels and args.output):
        parser.error('--replacements, --labels and --output are required')
    integrate(args.base.resolve(), [p.resolve() for p in args.replacements],
              args.labels.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
