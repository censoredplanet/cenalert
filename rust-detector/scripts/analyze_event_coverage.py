"""Analyze ensemble event coverage, calibration changes, and grouping outcomes."""
import argparse
import csv
import shlex
from pathlib import Path
from statistics import mean, pstdev

from analyze_paper_results import Analysis, event_text, markdown
from batch_ensemble_tuning import label_applies
from review_contextualization import matches, read, write, write_unmatched_notifications
from analyze_notification_outcomes import analyze as alert_quality, markdown as quality_markdown

CHOSEN = 'labels_ensemble/knee'
HEADING = '## Ensemble-only test events and calibration changes'


def ensemble_only(context, labels, first, last):
    """Match IF+LOF-only onset votes to labels and independently tuned baselines."""
    rows = {(r['country'], r['variant']): r for r in context['countries']
            if r['selection'] == 'knee'}
    cases, events = [], []
    for (country, variant), row in sorted(rows.items()):
        if variant != 'labels_ensemble':
            continue
        applicable = [l for l in labels if first <= l['date'] <= last and label_applies(l, country)]
        selected = []
        for anomaly in row['raw_anomalies']:
            vote = anomaly.get('ensemble_vote')
            if anomaly.get('sparse') or not vote or vote['chebyshev']['voted'] or not (
                    vote['isolation_forest']['voted'] and vote['lof']['voted']):
                continue
            matched = [l for l in applicable if matches(anomaly, l, context['country_proximity_days'])]
            selected.append(anomaly)
            cases.append(dict(country=country, start=anomaly['start'], end=anomaly['end'],
                impact=anomaly['impact'], matched=bool(matched),
                events='; '.join(f"{l['date']} {event_text(l)}" for l in matched)))
        for label in applicable:
            starts = [a['start'] for a in selected if matches(a, label, context['country_proximity_days'])]
            if label['scope'] != 'country' or not starts:
                continue
            entry = dict(country=country, date=label['date'], event=event_text(label), starts=';'.join(starts))
            for baseline in ('original', 'labels'):
                entry[baseline + '_detected'] = any(matches(a, label, context['country_proximity_days'])
                    for a in rows[country, baseline]['raw_anomalies'])
            events.append(entry)
    return dict(intervals=len(cases), matched_intervals=sum(c['matched'] for c in cases),
        country_labels=len(events), missed_by_original=sum(not e['original_detected'] for e in events),
        missed_by_labeled_single=sum(not e['labels_detected'] for e in events), cases=cases, events=events)


def calibration_changes(before, after):
    def selected(context):
        return {r['country']: r for r in context['countries']
                if r['variant'] == 'labels_ensemble' and r['selection'] == 'knee'}
    old, new = selected(before), selected(after)
    changes = []
    for country in sorted(old.keys() & new.keys()):
        a, b = (r['calibration']['matched_training_anomalies'] for r in (old[country], new[country]))
        if a != b and (a <= 1 or b <= 1):
            changes.append(dict(country=country, before=a, after=b,
                old_evidence=old[country]['calibration'], new_evidence=new[country]['calibration']))
    return changes


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def inspect_series(root, dashboard):
    output = []
    for country, when in [('NP', '2025-07-19'), ('SA', '2025-09-04'), ('RU', '2025-03-20')]:
        series = read(root/f'inputs/{country}-evaluation.json')
        current = {p['date']: p['rate'] for p in read(dashboard/f'{country}.json')}
        center = next(i for i, p in enumerate(series) if p['date'] == when)
        days = []
        for i in range(center-3, center+4):
            point = series[i]
            history = [p['rate'] for p in series[i-60:i]]
            average, std = mean(history), pstdev(history)
            days.append(dict(date=point['date'], value=point['rate'], dashboard_value=current[point['date']],
                preceding_60_day_mean=average, preceding_60_day_std=std,
                fixed_threshold=average+3*std, z_score=(point['rate']-average)/std if std else None))
        output.append(dict(country=country, label_date=when, days=days,
            local_dashboard_matches=all(abs(p['value']-p['dashboard_value']) < 1e-10 for p in days),
            largest_prior_values=sorted(series[center-60:center],key=lambda p:p['rate'],reverse=True)[:5]))
    return output


def prepare_diagnostic(root, destination, outcomes):
    """Export full-period feasibility inputs and a batch command."""
    countries = sorted({r['country'] for r in outcomes if r[CHOSEN] != 'retained'})
    destination.mkdir(parents=True, exist_ok=True)
    series = destination/'series'
    series.mkdir(exist_ok=True)
    config = read(root/'experiment.json')
    labels = [l for l in read(root/'inputs/labels.json') if l['date'] <= config['test_end']]
    write(destination/'labels.json', labels)
    for country in countries:
        write(series/f'{country}.json', read(root/f'inputs/{country}-evaluation.json'))
    repo = Path(__file__).resolve().parents[1]
    command = [str(repo/'.venv/bin/python'), 'scripts/batch_ensemble_tuning.py',
        '--series-directory', str(series), '--labels', str(destination/'labels.json'),
        '--rust-binary', str(root/'inputs/cenalert-detector'),
        '--output-directory', str(destination/'results'), '--work-directory', str(destination/'work'),
        '--countries', *countries, '--evaluations', '5000', '--population-size', '100',
        '--seed', '42', '--workers', '4', '--threads-per-tuning', '1',
        '--multi-country-threshold', '4',
        '--dataset-description', 'Full-period label feasibility check, 2011-01-01 through '+config['test_end']]
    (destination/'run.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(repo))+
        '\nexec '+shlex.join(command)+' "$@"\n')
    return countries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--previous-results', type=Path,
                        help='Optional prior results for comparing training calibration counts.')
    parser.add_argument('--minimum-group-countries', type=int, default=4)
    parser.add_argument('--diagnostic-directory', type=Path)
    args = parser.parse_args()
    root, cache = args.results.resolve(), args.cache.resolve()
    context_path = root / 'contextualization.json'
    context = read(context_path)
    previous_context = None
    if args.previous_results:
        previous_context = read(args.previous_results / 'contextualization.json')
        assert read(root/'inputs/training-labels.json') == read(args.previous_results/'inputs/training-labels.json')
    key = str(args.minimum_group_countries)
    if args.minimum_group_countries < 2:
        parser.error('minimum group size must be at least two')
    # Only grouping selection changes. Check that all saved replay inputs still match the cache.
    manifest = read(cache/'inputs.json')
    for relative, signature in manifest.items():
        if relative == 'contextualization':
            continue
        path = root/relative
        assert [path.stat().st_size, path.stat().st_mtime_ns] == signature, relative
    for summaries, groups in (('summary', 'grouping'), ('available_summary', 'available_grouping')):
        for name, summary in context[summaries].items():
            assert key in context[groups][name], (name, key)
            summary['minimum_group_countries'] = args.minimum_group_countries
    context['minimum_group_countries'] = args.minimum_group_countries
    write(context_path, context)
    manifest['contextualization'] = [str(context_path), context_path.stat().st_mtime_ns]
    write(cache/'inputs.json', manifest)
    analysis = Analysis(root, context_path, cache)
    references = [analysis.calibrate(r) for r in context['countries']]
    result = read(root/'analysis.json')
    result['notifications'], notifications = analysis.notifications(references)
    result['comparisons'], result['event_differences'] = analysis.comparisons(references)
    with (root/'event-review.csv').open() as stream:
        other_events = [r for r in csv.DictReader(stream) if r['analysis'] not in
                        ('notification', 'selection_difference', 'filtered_test_label')]
    write_csv(root/'event-review.csv', notifications + result['event_differences'] + other_events)
    result['alert_quality'] = alert_quality(analysis, result, 4)
    review = ensemble_only(context, analysis.labels, analysis.first_test, analysis.config['test_end'])
    changes = calibration_changes(previous_context, context) if previous_context else []
    result['submission_review'] = dict(ensemble_only=review, calibration_changes=changes)
    if args.diagnostic_directory:
        destination = args.diagnostic_directory.resolve()
        result['submission_review']['series_checks'] = inspect_series(root, destination/'dashboard')
        result['submission_review']['dashboard_downloaded_at'] = read(
            destination/'dashboard-manifest.json')['downloaded_at']
        result['submission_review']['diagnostic_countries'] = prepare_diagnostic(
            root, destination, result['alert_quality']['test_label_outcomes'])
    write(root/'analysis.json', result)
    write_unmatched_notifications(root/'unmatched-notifications.csv', context['available_grouping'][CHOSEN][key])
    write_csv(root/'iforest-lof-test-events.csv', review['events'])
    lines = [HEADING, '',
        f"The {review['intervals']} dense IF+LOF-only intervals include {review['matched_intervals']} intervals "
        f"matching country/global labels and cover {review['country_labels']} distinct country events. "
        f"Of these, {review['missed_by_original']} are missed by the unconstrained single detector and "
        f"{review['missed_by_labeled_single']} by the label-constrained single detector (knee selection).", '',
        'The comparison uses each single-detector configuration’s independently tuned parameters. '
        '`iforest-lof-test-events.csv` lists the country events.', '',
        'Calibration counts refer to detected, coalesced training anomalies matched by country or global labels. '
        'Each configuration calibrates its minimum impact from its own matched training anomalies.', '']
    for row in changes:
        lines.append(f"- {row['country']}: {row['before']} to {row['after']} explained training anomalies.")
    lines += ['', f"Grouping at {key} countries yields "
        f"{context['available_summary'][CHOSEN]['notifications'][key]} notifications. "
        '`unmatched-notifications.csv` lists their unmatched constituents; shared notification IDs identify one group.', '']
    if args.diagnostic_directory:
        lines += ['### Event-window checks', '',
            'Dashboard retrieval: '+result['submission_review']['dashboard_downloaded_at']+'. '
            'The seven-day windows below compare the experiment inputs with these current series.', '']
        for check in result['submission_review']['series_checks']:
            point = next(p for p in check['days'] if p['date'] == check['label_date'])
            lines.append(f"- {check['country']} {check['label_date']}: value {point['value']:.3f}; "
                         f"preceding-60-day 3σ threshold {point['fixed_threshold']:.3f}; z-score {point['z_score']:.3f}; "
                         f"Dashboard window agrees: {check['local_dashboard_matches']}.")
        lines += ['', 'Full-period feasibility inputs use 2011–August 2026 observations and all labels through '
                  'August 2026. Prepared countries: '+', '.join(result['submission_review']['diagnostic_countries'])+'.', '']
    (root/'analysis.md').write_text(markdown(result)+'\n\n'+quality_markdown(result['alert_quality'])+'\n\n'+'\n'.join(lines))
    print({k:v for k,v in review.items() if k not in ('cases','events')})
    print('Calibration changes:', [(r['country'],r['before'],r['after']) for r in changes])
    print('Exported event coverage, calibration counts, and grouped notifications.')


if __name__ == '__main__':
    main()
