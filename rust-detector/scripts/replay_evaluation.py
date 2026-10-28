"""Replay saved parameters on repaired test observations and updated test labels."""
import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repair_evaluation_dates import fingerprint
from review_contextualization import read, write
from run_experiment_matrix import evaluate_period, export_results, fixed_three_sigma


def replay(root, repairs, country, config):
    source = repairs/'series'/f'{country}.json'
    rows = [r for r in read(source) if r['date'] <= config['test_end']]
    old = read(root/f'inputs/{country}-evaluation.json')
    assert [r for r in rows if r['date'] <= config['train_end']] == [
        r for r in old if r['date'] <= config['train_end']]
    evaluation = repairs/'series'/f'{country}-evaluation.json'
    write(evaluation, rows)
    output = []
    for folder in sorted((root/f'runs/{config["seeds"][0]}/{country}').iterdir()):
        run = read(folder/'result.json')
        if run['status'] != 'completed':
            continue
        binary = root/'inputs'/('cenalert-detector' if 'ensemble' in run['variant'] else 'single-detector-binary')
        if not binary.exists():
            binary = root/'inputs/cenalert-detector'
        for selection in run['selections']:
            name = selection['selection']
            path = folder/f'{name}-detection.json'
            target = repairs/'replays'/path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if rows == old:
                result = read(path)
            elif name == 'fixed':
                result = fixed_three_sigma(rows, config['baseline_window'], run['training_maximum'])
            else:
                command = [str(binary), 'detect', '--algorithm', run['algorithm'],
                    '--input', str(evaluation), '--parameters', str(folder/f'{name}-parameters.json'),
                    '--reference-end', config['train_end']]
                result = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            before = read(path)
            assert [a for a in before['anomalies'] if a['end'] <= config['train_end']] == [
                a for a in result['anomalies'] if a['end'] <= config['train_end']], (country, run['variant'], name)
            write(target, result)
            output.append(dict(country=country, variant=run['variant'], selection=name,
                before=[a for a in before['anomalies'] if a['start'] > config['train_end']],
                after=[a for a in result['anomalies'] if a['start'] > config['train_end']]))
    return output


def apply(root, repairs, labels, config):
    audit = read(repairs/'data-quality.json')
    for country, expected in audit['source_sha256'].items():
        if fingerprint(root/f'inputs/{country}.json') != expected:
            raise ValueError(f'Source changed since repair preparation: {country}')
    if [l for l in labels if l['date'] <= config['train_end']] != read(root/'inputs/training-labels.json'):
        raise ValueError('Training labels changed; only test-label updates are supported')
    backup = repairs/'original'
    if backup.exists():
        raise ValueError('Repairs already applied or backup exists')
    # Validate all replacements before touching the published inputs.
    for country in config['countries']:
        for suffix in ('.json', '-evaluation.json'):
            before = read(root/f'inputs/{country}{suffix}')
            after = read(repairs/'series'/f'{country}{suffix}')
            if [r['date'] for r in before] != [r['date'] for r in after]:
                raise ValueError(f'Observation dates changed: {country}{suffix}')
            if any(a != b for a, b in zip(before, after) if a['date'] not in audit['dates']):
                raise ValueError(f'Unexpected observation change: {country}{suffix}')
    for path in sorted((root/'runs').glob('*/*/*/result.json')):
        run = read(path)
        if run['status'] != 'completed':
            continue
        for selection in run['selections']:
            detection = path.parent/f'{selection["selection"]}-detection.json'
            before, after = read(detection), read(repairs/'replays'/detection.relative_to(root))
            if [a for a in before['anomalies'] if a['end'] <= config['train_end']] != [
                    a for a in after['anomalies'] if a['end'] <= config['train_end']]:
                raise ValueError(f'Training detections changed: {detection}')
    backup.mkdir(parents=True)
    shutil.copy2(root/'input-summary.json', backup/'input-summary.json')
    for country in config['countries']:
        for suffix in ('.json', '-evaluation.json'):
            source = root/f'inputs/{country}{suffix}'
            target = backup/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            shutil.copy2(repairs/'series'/f'{country}{suffix}', source)
    shutil.copy2(root/'inputs/labels.json', backup/'inputs/labels.json')
    write(root/'inputs/labels.json', labels)
    details = read(root/'input-summary.json')
    for country, entry in details.items():
        entry['zero_observations'] = sum(r['rate'] == 0 for r in read(root/f'inputs/{country}.json'))
    write(root/'input-summary.json', details)
    runs = []
    for path in sorted((root/'runs').glob('*/*/*/result.json')):
        run = read(path)
        run['zero_observations'] = details[run['country']]['zero_observations']
        original = backup/path.relative_to(root)
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, original)
        if run['status'] == 'completed':
            for s in run['selections']:
                detection = path.parent/f'{s["selection"]}-detection.json'
                shutil.copy2(detection, backup/detection.relative_to(root))
                shutil.copy2(repairs/'replays'/detection.relative_to(root), detection)
                anomalies = read(detection)['anomalies']
                s.update(evaluate_period(anomalies, labels, run['country'], run['test_first_date'],
                                         run['test_last_date'], config['proximity_days']))
                for year in s['yearly']:
                    year.update(evaluate_period(anomalies, labels, run['country'],year['first_date'],
                                                year['last_date'], config['proximity_days']))
        write(path, run)
        runs.append(run)
    export_results(root, runs, config['proximity_days'])
    write(root/'data-quality.json', audit)
    integration = read(root/'integration.json')
    integration['evaluation_repairs'] = dict(dates=audit['dates'], method=audit['method'],
        training_unchanged=True, audit='data-quality.json')
    write(root/'integration.json', integration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--repairs', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root, repairs = args.results.resolve(), args.repairs.resolve()
    config = read(root/'experiment.json')
    if len(config['seeds']) != 1 or args.workers < 1:
        parser.error('Replay expects a single-seed experiment and a positive worker count')
    if args.apply:
        apply(root, repairs, read(repairs/'labels.json'), config)
        print('Applied test-data repairs and refreshed label metrics; saved parameters unchanged.')
        return
    comparisons = []
    with ThreadPoolExecutor(args.workers) as pool:
        for country, output in zip(config['countries'], pool.map(
                lambda c: replay(root, repairs, c, config), config['countries'])):
            comparisons.extend(output)
            print(country, 'fixed-parameter replay complete', flush=True)
    write(repairs/'detection-comparison.json', comparisons)


if __name__ == '__main__':
    main()
