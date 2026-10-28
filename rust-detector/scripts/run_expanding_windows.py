"""Prepare and run expanding-window Z-Score evaluation independently of paper results."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from batch_ensemble_tuning import Config, detect, write_csv
import run_experiment_matrix as matrix

VARIANTS = ['original', 'labels']
SELECTIONS = ['knee', 'leftmost']
WINDOWS = [('2018-12-31', '2020-12-31'), ('2020-12-31', '2022-12-31'),
           ('2022-12-31', '2024-12-31'), ('2024-12-31', '2026-08-31')]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_name(cutoff):
    return 'train-through-' + cutoff[:4]


def matrix_arguments(root, plan, cutoff, end):
    return ['--series-directory', str(root / 'series'), '--labels', str(root / 'labels.json'),
            '--rust-binary', str(root / 'cenalert-detector'),
            '--output-directory', str(root / window_name(cutoff)),
            '--variants', *VARIANTS, '--selections', *SELECTIONS,
            '--train-end', cutoff, '--test-end', end,
            '--evaluations', str(plan['evaluations']), '--population-size', str(plan['population_size']),
            '--seeds', str(plan['seed']), '--workers', str(plan['workers']),
            '--threads-per-tuning', '1', '--proximity-days', str(plan['proximity_days'])]


def prepare(source, root, binary, workers):
    source, root, binary = source.resolve(), root.resolve(), binary.resolve()
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('Use a separate experiment directory outside the source results')
    if root.exists():
        raise FileExistsError(f'{root} already exists')
    if workers < 1:
        raise ValueError('Workers must be positive')
    config = read(source / 'experiment.json')
    if config['test_end'] != WINDOWS[-1][1]:
        raise ValueError('Expected source series through 2026-08-31')
    binary.read_bytes()
    labels = read(source / 'inputs/labels.json')
    series = {}
    for country in config['countries']:
        rows = read(source / 'inputs' / f'{country}.json')
        if any(r['rate'] > 0 for r in rows if r['date'] <= WINDOWS[-1][0]):
            series[country] = rows
    if not series:
        raise ValueError('No countries with nonzero training observations')
    root.mkdir(parents=True)
    (root / 'series').mkdir()
    (root / 'code').mkdir()
    for country, rows in series.items():
        matrix.write_json(root / 'series' / f'{country}.json', rows)
    matrix.write_json(root / 'labels.json', labels)
    shutil.copy2(binary, root / 'cenalert-detector')
    for name in ('run_expanding_windows.py', 'run_experiment_matrix.py', 'batch_ensemble_tuning.py'):
        shutil.copy2(Path(__file__).with_name(name), root / 'code' / name)
    plan = dict(source_results=str(source), source_binary=str(binary),
                windows=WINDOWS, variants=VARIANTS, selections=SELECTIONS,
                countries=sorted(series), evaluations=5000, population_size=100,
                seed=42, workers=workers, proximity_days=config['proximity_days'])
    for cutoff, end in WINDOWS:
        # Preparation calls only the snapshot function, never the tuning workflow.
        with_arguments = [str(root / 'code/run_experiment_matrix.py'),
                          *matrix_arguments(root, plan, cutoff, end), '--prepare-only']
        subprocess.run([sys.executable, '-B', *with_arguments], check=True)
    plan['snapshot_sha256'] = {
        str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts
    }
    matrix.write_json(root / 'plan.json', plan)
    print(f'Prepared {len(WINDOWS)} windows, {len(series)} countries; no tuning started.')


def verify_snapshot(root, plan):
    for name, expected in plan['snapshot_sha256'].items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Prepared input changed: {path}')


def previous_result(root, plan, cutoff, end, previous, country, variant, selection):
    current_dir = root / window_name(cutoff)
    old_dir = root / window_name(previous) / 'runs' / str(plan['seed']) / country / variant
    source = read(old_dir / 'result.json')
    result = dict(country=country, variant=variant, selection=selection,
                  parameter_train_end=previous, train_end=cutoff, test_end=end,
                  model='previous', status=source['status'])
    if source['status'] != 'completed':
        return result
    details = read(current_dir / 'input-summary.json')[country]
    if not details['test_observations']:
        result['status'] = 'skipped_missing_period'
        return result
    output = current_dir / 'previous-parameters' / country / variant
    output.mkdir(parents=True, exist_ok=True)
    cached = output / f'{selection}-result.json'
    if cached.exists():
        return read(cached)
    config = Config(root / 'labels.json', root / 'cenalert-detector', current_dir,
                    plan['evaluations'], plan['population_size'], plan['proximity_days'],
                    1, 1, plan['seed'], algorithm='chebyshev')
    detected = detect(config, current_dir / 'inputs' / f'{country}-evaluation.json',
                      old_dir / f'{selection}-parameters.json', previous)
    matrix.write_json(output / f'{selection}-detection.json', detected)
    metrics = matrix.evaluate_period(detected['anomalies'], read(root / 'labels.json'), country,
                                     details['test_first_date'], details['test_last_date'],
                                     plan['proximity_days'])
    result.update(metrics)
    matrix.write_json(cached, result)
    return result


def summarize(rows):
    summaries = []
    for cutoff, end in WINDOWS:
        for variant in VARIANTS:
            for selection in SELECTIONS:
                subset = [r for r in rows if (r['train_end'], r['variant'], r['selection']) ==
                          (cutoff, variant, selection)]
                completed = {model: [r for r in subset if r['model'] == model and r['status'] == 'completed']
                             for model in ('retuned', 'previous')}
                common = {r['country'] for r in completed['retuned']} & {
                    r['country'] for r in completed['previous']}
                cohorts = [('all_completed', 'retuned', completed['retuned'])]
                if any(r['model'] == 'previous' for r in subset):
                    cohorts += [('paired', model, [r for r in records if r['country'] in common])
                                for model, records in completed.items()]
                for cohort, model, records in cohorts:
                    total = sum(r['country_labels_total'] for r in records)
                    covered = sum(r['country_labels_covered'] for r in records)
                    summaries.append(dict(train_end=cutoff,
                        test_start=str(date.fromisoformat(cutoff) + timedelta(days=1)), test_end=end,
                        variant=variant, selection=selection, cohort=cohort, model=model,
                        countries=len(records), anomalies=sum(r['anomaly_count'] for r in records),
                        labels_covered=covered, labels_total=total,
                        coverage=covered/total if total else None))
    return summaries


def run(root):
    root = root.resolve()
    plan = read(root / 'plan.json')
    verify_snapshot(root, plan)
    for cutoff, end in plan['windows']:
        print(f'Training through {cutoff}; testing through {end}', flush=True)
        subprocess.run([sys.executable, str(root / 'code/run_experiment_matrix.py'),
                        *matrix_arguments(root, plan, cutoff, end), '--resume'], check=True)
    rows = []
    for index, (cutoff, end) in enumerate(plan['windows']):
        for country in plan['countries']:
            for variant in VARIANTS:
                result = read(root / window_name(cutoff) / 'runs' / str(plan['seed']) /
                              country / variant / 'result.json')
                for selection in SELECTIONS:
                    metrics = next((s for s in result['selections'] if s['selection'] == selection), {})
                    rows.append(dict(country=country, variant=variant, selection=selection,
                        parameter_train_end=cutoff, train_end=cutoff, test_end=end,
                        model='retuned', status=result['status'], **{
                            k: metrics[k] for k in ('anomaly_count', 'country_labels_covered', 'country_labels_total')
                            if k in metrics}))
        if index:
            previous = plan['windows'][index-1][0]
            tasks = [(country, variant, selection) for country in plan['countries']
                     for variant in VARIANTS for selection in SELECTIONS]
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                rows.extend(pool.map(lambda task: previous_result(root, plan, cutoff, end, previous, *task), tasks))
    fields = ['train_end', 'test_end', 'parameter_train_end', 'country', 'variant', 'selection',
              'model', 'status', 'anomaly_count', 'country_labels_covered', 'country_labels_total']
    write_csv(root / 'country-results.csv', fields, [{k: r.get(k) for k in fields} for r in rows])
    summary = summarize(rows)
    write_csv(root / 'summary.csv', list(summary[0]), summary)
    print(f'Finished. Summary: {root / "summary.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--source-results', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--rust-binary', type=Path, required=True)
    preparation.add_argument('--workers', type=int, default=4)
    execution = commands.add_parser('run')
    execution.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source_results, args.output, args.rust_binary, args.workers)
    else:
        run(args.directory)


if __name__ == '__main__':
    main()
