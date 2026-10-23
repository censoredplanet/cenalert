"""Attribute label coverage to votes at saved ensemble anomaly onsets."""
import argparse
import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from batch_ensemble_tuning import label_applies, required_labels
from review_contextualization import matches, read

MEMBERS = ('chebyshev', 'isolation_forest', 'lof')


def export(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    config = read(root / 'experiment.json')
    labels = read(root / 'inputs/labels.json')
    cutoff = config['train_end']
    first_test = str(date.fromisoformat(cutoff) + timedelta(days=1))
    proximity = config['proximity_days']
    summary = defaultdict(lambda: defaultdict(int))
    events, costs = [], []
    for path in sorted((root / 'runs').glob('*/*/*/result.json')):
        run = read(path)
        if run['status'] != 'completed' or run['variant'] == 'fixed_3sigma':
            continue
        country, variant = run['country'], run['variant']
        for selected in run['selections']:
            selection = selected['selection']
            detection = read(path.parent / f'{selection}-detection.json')['anomalies']
            cost = dict(country=country, seed=run['seed'], configuration=variant, selection=selection)
            for period, first, last in [('training', run['first_date'], cutoff),
                                        ('test', first_test, config['test_end'])]:
                anomalies = [a for a in detection if first <= a['start'] <= last]
                required = required_labels(labels, country, first, last, 0)
                covered = [l for l in required if any(matches(a, l, proximity) for a in anomalies)]
                cost.update({period + '_anomalies': len(anomalies),
                             period + '_labels_total': len(required),
                             period + '_labels_covered': len(covered)})
                if 'ensemble' not in variant:
                    continue
                applicable = [l for l in labels if first <= l['date'] <= last and label_applies(l, country)]
                subsets = {'ensemble': anomalies,
                           'croston': [a for a in anomalies if a.get('sparse', False)]}
                for member in MEMBERS:
                    subsets[member] = [a for a in anomalies if a.get('ensemble_vote', {}).get(member, {}).get('voted', False)]
                subsets['iforest_lof_without_chebyshev'] = [a for a in subsets['isolation_forest']
                    if a['ensemble_vote']['lof']['voted'] and not a['ensemble_vote']['chebyshev']['voted']]
                for member, accepted in subsets.items():
                    row = summary[variant, selection, period, member]
                    row['countries'] += 1
                    row['country_labels_total'] += len(required)
                    row['accepted_anomalies'] += len(accepted)
                    row['anomalies_matching_any_label'] += sum(any(matches(a, l, proximity) for l in applicable) for a in accepted)
                    row['country_labels_supported'] += sum(any(matches(a, l, proximity) for a in accepted) for l in required)
                for label in required:
                    matched = [a for a in anomalies if matches(a, label, proximity)]
                    events.append(dict(country=country, seed=run['seed'], configuration=variant,
                        selection=selection, period=period, label_date=label['date'],
                        event=label.get('title') or label.get('description', ''),
                        ensemble_detected=bool(matched),
                        matched_starts=';'.join(a['start'] for a in matched),
                        croston_supported=any(a.get('sparse', False) for a in matched),
                        **{member + '_supported': any(a.get('ensemble_vote', {}).get(member, {}).get('voted', False)
                                                      for a in matched) for member in MEMBERS}))
            costs.append(cost)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(configuration=v, selection=s, period=p, member=m, **values)
            for (v, s, p, m), values in sorted(summary.items())]
    export(output / 'member-contributions.csv', rows)
    export(output / 'label-contributions.csv', events)
    export(output / 'constraint-cost.csv', costs)
    print(f'Analyzed {len(costs)} selected configurations and {len(events)} label/configuration pairs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
