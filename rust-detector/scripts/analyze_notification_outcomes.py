"""Analyze label coverage, severity, and regional notification volume."""
import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from statistics import median

from analyze_paper_results import Analysis, event_text, member_parameters
from batch_ensemble_tuning import required_labels
from review_contextualization import grouping_key, matches, read, write


# Access Now's six-region convention, including Türkiye and Sudan in MENA.
# Countries absent from STOP are assigned geographically to the same regions.
REGION_COUNTRIES = {
    'Africa': 'BF BI BJ CD CM ET GA GN GQ KE ML MU NG SN SO SS SZ TG TZ UG ZM ZW',
    'APAC': 'AF BD CN HK ID IN KH LK MM MY NP PH PK TH VN',
    'Europe': 'ES GB IT',
    'EECA': 'AM AZ BY KG KZ RU TJ TM UA UZ',
    'LAC': 'BR CU EC SR VE',
    'MENA': 'AE DZ EG IQ IR JO LB LY MA OM QA SA SD TN TR YE',
}
REGIONS = {c: r for r, countries in REGION_COUNTRIES.items() for c in countries.split()}
REGION_SOURCES = [
    'https://www.accessnow.org/guide/shutdown-tracker-optimization-project/',
    'https://www.accessnow.org/press-release/mena-keepiton-internet-shutdowns-2023-en/',
    'https://www.accessnow.org/press-release/keepiton-internet-shutdowns-2022-eeca/',
]
OUTCOMES_HEADING = '## Event coverage and notification volume'


def calibration_probability(reference):
    matched = {i for r in reference['references'] for i in r['labels']}
    applicable = set(reference['applicable_labels'])
    assert matched <= applicable
    return len(matched) / len(applicable) if applicable else 0.0


def label_status(row, label, proximity):
    if any(matches(a, label, proximity) for a in row['retained_anomalies']):
        return 'retained'
    if any(matches(a, label, proximity) for a in row['raw_anomalies']):
        return 'filtered'
    return 'not_detected'


def test_label_outcomes(analysis):
    countries = sorted({r['country'] for r in analysis.context['countries']})
    keys = list(analysis.context['summary'])
    by_key = {(r['country'], f"{r['variant']}/{r['selection']}"): r
              for r in analysis.context['countries']}
    rows = []
    for country in countries:
        labels = required_labels(analysis.labels, country, analysis.first_test,
                                 analysis.config['test_end'], 0)
        for label in labels:
            row = dict(country=country, date=label['date'], event=event_text(label))
            for key in keys:
                record = by_key.get((country, key))
                row[key] = label_status(record, label, analysis.proximity) if record else 'no_solution'
            record = by_key.get((country, 'labels_ensemble/knee'))
            ensemble = record['raw_anomalies'] if record else []
            nearest = min(ensemble, key=lambda a:abs((date.fromisoformat(a['start'])-
                          date.fromisoformat(label['date'])).days), default=None)
            row['ensemble_knee_nearest_start'] = nearest['start'] if nearest else ''
            row['ensemble_knee_start_offset_days'] = ((date.fromisoformat(nearest['start'])-
                date.fromisoformat(label['date'])).days) if nearest else ''
            row['ensemble_knee_containing_interval'] = ';'.join(
                f"{a['start']}..{a['end']}" for a in ensemble if a['start']<=label['date']<=a['end'])
            cached = read(analysis.cache / f'{country}-labels_ensemble-knee-members.json') if record else {'events': []}
            row['ensemble_knee_rejected_single_votes'] = ';'.join(sorted({
                r['member'] for r in cached['events'] if r['analysis']=='rejected_single_vote'
                and r['period']=='test' and r['date']==label['date'] and r['event']==event_text(label)}))
            rows.append(row)
    for key in keys:
        totals = Counter(r[key] for r in rows)
        expected = analysis.context['available_summary'][key]
        assert totals['retained'] == expected['retained_covered']
        assert totals['retained'] + totals['filtered'] == expected['raw_covered']
        assert len(rows) - totals['no_solution'] == expected['labels_total']
    return rows


def month_range(first, last):
    year, month = map(int, first[:7].split('-'))
    result = []
    while f'{year:04d}-{month:02d}' <= last[:7]:
        result.append(f'{year:04d}-{month:02d}')
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def region_months(notifications, months):
    counts = Counter()
    for row in notifications:
        assert row['date'][:7] in months
        for region in {REGIONS[c] for c in row['country'].split(';')}:
            counts[region, row['date'][:7]] += 1
    return [dict(region=region, month=month, notifications=counts[region, month])
            for region in REGION_COUNTRIES for month in months]


def iforest_country(analysis, row):
    """Measure detection sensitivity to two Isolation Forest threshold floors."""
    country = row['country']
    folder = analysis.folder(country, 'labels_ensemble')
    parameters = read(folder / 'knee-parameters.json')
    run = read(folder / 'result.json')
    maximum = max(p['rate'] for p in read(analysis.root / f'inputs/{country}-evaluation.json'))
    p = member_parameters(parameters, 'isolation_forest', run['training_maximum'], maximum)
    cache = analysis.cache / f'{country}-labels_ensemble-knee-iforest-diagnostic.json'
    signature = dict(parameters=p, train_end=analysis.cutoff, test_end=analysis.config['test_end'])
    if cache.exists():
        saved = read(cache)
        if saved['settings'] == signature:
            return saved['rows']
    required = required_labels(analysis.labels, country, analysis.first_test,
                               analysis.config['test_end'], 0)
    output = []
    for floor in (None, .65, .70):
        chosen = dict(p, min_score=max(p['min_score'], floor or 0))
        detected = analysis.detect(country, 'isolation_forest', chosen)['anomalies']
        anomalies = [a for a in detected if analysis.first_test <= a['start'] <= analysis.config['test_end']]
        dense = [a for a in anomalies if not a.get('sparse', False)]
        durations = [(date.fromisoformat(a['end']) - date.fromisoformat(a['start'])).days + 1 for a in dense]
        scores = [a['score'] for a in dense if a['score'] is not None]
        output.append(dict(country=country, threshold_floor=floor, window=p['window'],
            selected_threshold=p['min_score'], applied_threshold=chosen['min_score'],
            anomalies=len(anomalies), dense=len(dense), sparse=len(anomalies)-len(dense),
            dense_one_day=sum(d == 1 for d in durations), dense_durations=durations,
            dense_onset_scores=scores,
            test_labels_covered=sum(any(matches(a,l,analysis.proximity) for a in anomalies) for l in required)))
        if floor is None:
            previous = read(analysis.cache / f'{country}-labels_ensemble-knee-members.json')['summary']
            previous = next(r for r in previous if r['member']=='isolation_forest'
                            and r['period']=='test' and r['mode']=='standalone_shared_parameters')
            assert len(anomalies) == previous['anomalies']
            assert len(dense) == previous['dense_anomalies']
    write(cache, dict(settings=signature, rows=output))
    return output


def analyze(analysis, previous, workers):
    references = [analysis.calibrate(r) for r in analysis.context['countries']]
    _, notifications = analysis.notifications(references)
    chosen = [r for r in notifications if r['configuration']=='labels_ensemble' and r['selection']=='knee']
    severity = []
    for category in ('restriction_related', 'other_explained', 'unexplained'):
        levels = Counter(r['severity'] for r in chosen if r['category']==category)
        severity.append(dict(category=category, total=sum(levels.values()),
                             **{level:levels[level] for level in ('low','medium','high','very_high')}))
    refs = [r for r in references if r['variant']=='labels_ensemble' and r['selection']=='knee']
    calibration = []
    for ref in refs:
        pool = ref['applicable_labels']
        matched = {i for r in ref['references'] for i in r['labels']}
        calibration.append(dict(country=ref['country'], applicable_labels=len(pool),
            global_labels=sum(analysis.train_labels[i]['scope']=='global' for i in pool),
            matched_labels=len(matched), probability=calibration_probability(ref)))
    sampling = next(r for r in previous['sampling']['summary'] if r['configuration']=='labels_ensemble'
                    and r['selection']=='knee' and r['mode']=='one_per_country')
    outcomes = test_label_outcomes(analysis)
    months = month_range(analysis.first_test, analysis.config['test_end'])
    manageability = []
    for key, summary in analysis.context['available_summary'].items():
        variant, selection = key.split('/')
        entries = [r for r in notifications if r['configuration']==variant and r['selection']==selection]
        region_rows = region_months(entries, months)
        regions = []
        for region in REGION_COUNTRIES:
            values = [r['notifications'] for r in region_rows if r['region']==region]
            regions.append(dict(region=region, total=sum(values), monthly_mean=sum(values)/len(months),
                monthly_median=median(values), monthly_max=max(values),
                trailing_months={str(n):sum(values[-n:]) for n in (1,3,6,12,len(months))}))
        manageability.append(dict(configuration=variant,selection=selection,
            raw_anomalies=summary['raw'],notifications=len(entries),
            monthly_mean=len(entries)/len(months),raw_monthly_mean=summary['raw']/len(months),
            regions=regions,region_months=region_rows))
    tasks = [r for r in analysis.context['countries'] if r['variant']=='labels_ensemble' and r['selection']=='knee']
    diagnostic = []
    with ThreadPoolExecutor(workers) as pool:
        for n, rows in enumerate(pool.map(lambda r:iforest_country(analysis,r),tasks),1):
            diagnostic.extend(rows)
            if n%20==0: print(f'Fixed-parameter iForest diagnostics: {n}/{len(tasks)}',flush=True)
    totals = []
    for floor in (None,.65,.70):
        rows = [r for r in diagnostic if r['threshold_floor']==floor]
        durations = [d for r in rows for d in r['dense_durations']]
        totals.append(dict(threshold_floor=floor, **{k:sum(r[k] for r in rows)
            for k in ('anomalies','dense','sparse','dense_one_day','test_labels_covered')},
            median_dense_duration=median(durations) if durations else None))
    return dict(minimum_group_countries=int(grouping_key(analysis.context)),
        calibration=dict(observed=sampling['calibrated_countries'],
                    expected=sum(r['probability'] for r in calibration), countries=calibration),
        severity_by_category=severity,test_label_outcomes=outcomes,
        region_mapping=REGIONS,region_sources=REGION_SOURCES,
        regional_counting='A cross-country notification counts once in every represented region; once globally.',
        months=months,manageability=manageability,
        original_paper=dict(months=24, notifications=159, monthly_mean=159/24,
            regions=dict(zip(REGION_COUNTRIES,(21,43,14,26,12,43)))),
        iforest=dict(summary=totals,countries=diagnostic),
        low_medium_restrictions=[r for r in chosen if r['category']=='restriction_related'
                                 and r['severity'] in ('low', 'medium')],
        grouping_sensitivity=grouping_sensitivity(analysis.context))


def grouping_sensitivity(context):
    output = []
    for threshold, grouping in context['available_grouping']['labels_ensemble/knee'].items():
        groups = []
        for group, annotated in zip(grouping['grouped_notifications'],
                                    grouping['label_matching']['groups'], strict=True):
            global_events = {}
            for member in annotated['members']:
                for label in member['labels']:
                    if label['scope'] == 'global':
                        key = (label['date'], event_text(label))
                        global_events.setdefault(key, set()).add(member['country'])
            groups.append(dict(date=group['date'], countries=group['countries'],
                anomalies=len(group['anomalies']),
                global_matches=[dict(date=d, event=e, countries=sorted(c))
                                for (d,e),c in sorted(global_events.items())]))
        output.append(dict(minimum_countries=int(threshold),
            notifications=grouping['label_matching']['total'],
            matched=grouping['label_matching']['matched'], groups=groups))
    return sorted(output, key=lambda r:r['minimum_countries'])


def markdown(result):
    calibration = result['calibration']['countries']
    global_counts = sorted({r['global_labels'] for r in calibration})
    ae = next((r for r in calibration if r['country']=='AE'), None)
    lines = [OUTCOMES_HEADING, '', '### Calibration event', '',
        'A calibration event is a coalesced training anomaly matched by an available training label. '
        'The minimum impact over these matches calibrates the country’s impact filter. '
        'The one-label condition draws uniformly from applicable country and global labels. '
        'An unmatched draw leaves impact filtering disabled.', '',
        f"For constrained ensemble/knee, the expected number of calibrated countries is {result['calibration']['expected']:.2f}; "
        f"the 100 repetitions produced {result['calibration']['observed']['mean']:.2f}. "
        f"The number of global labels in each country's candidate pool is {global_counts}. "
        'Most global labels do not match a training anomaly in a particular country. '
        + (f"For example, AE has {ae['applicable_labels']} applicable labels, of which {ae['matched_labels']} "
           f"match its coalesced training anomalies ({ae['matched_labels']}/{ae['applicable_labels']} probability)." if ae else ''), '',
        '### Severity by event category', '',
        '| Category | Low | Medium | High | Very high | Total |', '|---|---:|---:|---:|---:|---:|']
    for r in result['severity_by_category']:
        lines.append('| '+' | '.join(str(r[k]) for k in ('category','low','medium','high','very_high','total'))+' |')
    categories = {r['category']: r for r in result['severity_by_category']}
    restriction, other, unknown = (categories[k] for k in ('restriction_related','other_explained','unexplained'))
    high = lambda r:r['high']+r['very_high']
    lines += ['', f"High/very-high priority retains {high(restriction)}/{restriction['total']} restriction-related "
        f"notifications and {high(other)}/{other['total']} other explained notifications, "
        f"alongside {high(unknown)} unexplained notifications.", '',
        'Low/medium restriction-related notifications:', '']
    for r in result['low_medium_restrictions']:
        lines.append(f"- {r['country']} {r['date']} ({r['severity']}): {r['event']}")
    lines += ['', '### Cross-country grouping', '',
              '| Minimum countries | Notifications | Matched | Groups |', '|---|---:|---:|---:|']
    for r in result['grouping_sensitivity']:
        lines.append(f"| {r['minimum_countries']} | {r['notifications']} | {r['matched']} | {len(r['groups'])} |")
    for r in result['grouping_sensitivity']:
        if r['minimum_countries'] != result.get('minimum_group_countries', 5):continue
        for g in r['groups']:
            events = '; '.join(f"{e['event']} ({len(e['countries'])} countries)" for e in g['global_matches'])
            lines.append(f"- {g['date']}: {', '.join(g['countries'])}. Global label matches: {events or 'none'}.")
    lines += ['',
        '### Missed test events', '',
        '`test-label-outcomes.csv` records country labels across every configuration and selection. '
        '`not_detected` means no raw anomaly start matches; `filtered` means a raw match was removed by '
        'contextualization; `retained` means a matching candidate survives. Matching uses each configured '
        'country applicability and the three-day window.', '',
        'Constrained ensemble/knee misses these labels before contextualization:', '']
    for r in result['test_label_outcomes']:
        if r['labels_ensemble/knee']=='not_detected':
            alternatives=[k for k,v in r.items() if '/' in k and v=='retained']
            lines.append(f"- {r['country']} {r['date']} — {r['event']}. Retained by: {', '.join(alternatives) or 'none'}.")
    lines += ['', '### Isolation Forest', '',
        'Standalone members use the ensemble-selected parameters and their own evolving histories. '
        'The iForest diagnostic repeats detection with all other parameters fixed, raising its threshold '
        'floor to 0.65 or 0.70.', '',
        '| Threshold floor | All anomalies | Dense | Sparse | One-day dense | Test labels |',
        '|---|---:|---:|---:|---:|---:|']
    for r in result['iforest']['summary']:
        values=[r['threshold_floor'] or 'selected',*[r[k] for k in ('anomalies','dense','sparse','dense_one_day','test_labels_covered')]]
        lines.append('| '+' | '.join(map(str,values))+' |')
    original=[r for r in result['iforest']['countries'] if r['threshold_floor'] is None]
    low=[r for r in original if r['selected_threshold']<.62]
    lines += ['', f"{len(low)}/{len(original)} selected iForest thresholds are below 0.62; these countries account for "
        f"{sum(r['dense'] for r in low)}/{sum(r['dense'] for r in original)} dense detections. The five largest contributors are "
        + ', '.join(f"{r['country']} ({r['dense']})" for r in sorted(original,key=lambda r:r['dense'],reverse=True)[:5])+'.', '',
        '### Regional notification volume', '',
        'January 2025–August 2026 spans 20 months. Regional assignments use the six Access Now regions; '
        'Türkiye and Sudan are assigned to MENA. A grouped notification counts once per represented region '
        'and once in the global total. Thus regional totals can exceed the global total.', '',
        '| Configuration | Selection | Raw/month | Notifications/month |', '|---|---|---:|---:|']
    for r in result['manageability']:
        lines.append(f"| {r['configuration']} | {r['selection']} | {r['raw_monthly_mean']:.2f} | {r['monthly_mean']:.2f} |")
    selected=next(r for r in result['manageability'] if r['configuration']=='labels_ensemble' and r['selection']=='knee')
    lines += ['', 'Constrained ensemble/knee regional workload:', '',
        '| Region | Notifications | Mean/month | Median/month | Maximum/month |', '|---|---:|---:|---:|---:|']
    for r in selected['regions']:
        lines.append(f"| {r['region']} | {r['total']} | {r['monthly_mean']:.2f} | {r['monthly_median']:g} | {r['monthly_max']} |")
    lines += ['', f"The constrained ensemble yields {selected['notifications']} notifications over "
        f"{len(result['months'])} months ({selected['monthly_mean']:.2f}/month). "
        f"Contextualization reduces its workload from {selected['raw_monthly_mean']:.2f} "
        f"to {selected['monthly_mean']:.2f}/month.", '',
        'Regional classification sources: '+', '.join(result['region_sources'])+'.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1: parser.error('--workers must be positive')
    root=args.results.resolve()
    analysis=Analysis(root,root/'contextualization.json',args.cache.resolve())
    previous=read(root/'analysis.json')
    result=analyze(analysis,previous,args.workers)
    previous['alert_quality']=result
    write(root/'analysis.json',previous)
    with (root/'test-label-outcomes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['test_label_outcomes'][0]))
        writer.writeheader();writer.writerows(result['test_label_outcomes'])
    path=root/'analysis.md'
    text=path.read_text().split(OUTCOMES_HEADING)[0].split('## Alert quality and manageability')[0].rstrip()
    path.write_text(text+'\n\n'+markdown(result))
    print('Updated analysis.json, analysis.md, and test-label-outcomes.csv',flush=True)


if __name__=='__main__': main()
