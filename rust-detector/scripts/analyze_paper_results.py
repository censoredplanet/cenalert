"""Analyze fixed temporal detections, member ablations, and label subsampling."""
import argparse
import csv
import json
import math
import random
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median

from batch_ensemble_tuning import label_applies, required_labels
from review_contextualization import coverage, filter_output, matches, read, write
from review_event_outcomes import quantile
from review_member_contributions import analyze as accepted_member_analysis

MEMBERS = ('chebyshev', 'isolation_forest', 'lof')
FRACTIONS = (0, .01, .05, .1, .25, .5, .75, 1)
RESTRICTION_EVENTS = {
    ('BY','2025-01-09'), ('BY','2025-01-25'), ('CD','2025-02-01'),
    ('DZ','2025-06-14'), ('EG','2025-03-19'), ('GA','2026-02-17'),
    ('GN','2026-04-16'), ('ID','2025-08-30'), ('IR','2025-06-13'),
    ('IR','2025-11-15'), ('IR','2026-02-28'), ('JO','2025-09-10'),
    ('JO','2025-10-06'), ('KE','2025-06-25'), ('MM','2025-08-14'),
    ('NP','2025-07-19'), ('OM','2025-05-07'), ('OM','2026-08-13'),
    ('RU','2025-06-09'), ('RU','2025-07-09'), ('RU','2025-12-03'),
    ('SS','2025-01-22'), ('TM','2025-03-29'), ('TM','2025-07-21'),
    ('TR','2025-03-19'), ('TR','2025-09-08'), ('TR','2026-04-15'),
    ('VE','2025-01-08'), ('VN','2025-05-25'),
}


def member_parameters(parameters, member, training_maximum, series_maximum):
    result = {k: parameters[k] for k in ('window', 'min_residual_ratio', 'efficiency')}
    if member == 'chebyshev':
        result.update(z=parameters['chebyshev_z'], k=parameters['chebyshev_k'])
    else:
        result['min_score'] = parameters['isolation_forest_min_score' if member == 'isolation_forest' else 'lof_min_score']
        # The saved CLI exposes reference-end only for Chebyshev and ensemble.
        # Preserve the same absolute sparse threshold in the standalone scorer.
        result['min_residual_ratio'] *= training_maximum / series_maximum
    return result


def rejected_episodes(annotated, member):
    result = []
    for point in annotated:
        votes = point.get('ensemble_vote')
        rejected = (votes is not None and votes[member]['voted'] and
                    votes['vote_count'] == 1 and not point['anomaly'])
        if not rejected:
            continue
        if result and (date.fromisoformat(point['date']) - date.fromisoformat(result[-1]['end'])).days == 1:
            result[-1]['end'] = point['date']
        else:
            result.append(dict(start=point['date'], end=point['date']))
    return result


def sampling_threshold(references, selected):
    return min((r['impact'] for r in references if selected.intersection(r['labels'])), default=None)


def kept(impact, threshold, high):
    return threshold is None or impact >= threshold or (high is not None and impact >= high)


def event_text(label):
    return label.get('title') or label.get('description', '')


def label_kind(label):
    """Group the reviewed test labels by their recorded event descriptions."""
    category = label.get('metadata', {}).get('category')
    text = event_text(label).lower()
    if label['scope'] == 'global' or category in ('other', 'vpn_discussion'):
        return 'other_explained'
    if 'reaction to us tiktok ban' in text or 'restrictions never materialized' in text:
        return 'other_explained'
    if category in ('filtering','shutdowns') or (label.get('country_code'),label['date']) in RESTRICTION_EVENTS:
        return 'restriction_related'
    raise ValueError(f'Unclassified notification label: {label}')


def severity(impact, reference):
    if not reference:
        return 'unavailable'
    boundaries = [quantile(reference, p) for p in (.25, .5, .75)]
    return ('low', 'medium', 'high', 'very_high')[sum(impact >= b for b in boundaries)]


class Analysis:
    def __init__(self, root, context, cache):
        self.root, self.context, self.cache = root, read(context), cache
        self.config = read(root / 'experiment.json')
        self.labels = read(root / 'inputs/labels.json')
        self.train_labels = read(root / 'inputs/training-labels.json')
        self.cutoff = self.config['train_end']
        self.first_test = str(date.fromisoformat(self.cutoff) + timedelta(days=1))
        self.binary = root / 'inputs/cenalert-detector'
        self.seed, = self.config['seeds']
        self.proximity = self.config['proximity_days']
        cache.mkdir(parents=True, exist_ok=True)

    def folder(self, country, variant):
        return self.root / f'runs/{self.seed}/{country}/{variant}'

    def calibrate(self, row):
        key = f"{row['country']}-{row['variant']}-{row['selection']}"
        cached = self.cache / (key + '-calibration.json')
        if cached.exists():
            return read(cached)
        folder = self.folder(row['country'], row['variant'])
        anomalies = read(folder / f"{row['selection']}-detection.json")['anomalies']
        training = [a for a in anomalies if a['end'] <= self.cutoff]
        with tempfile.TemporaryDirectory() as temporary:
            coalesced, threshold = filter_output(self.binary, row['country'], training,
                self.root / 'inputs/training-labels.json', self.proximity, 1, Path(temporary))
        assert threshold == row['threshold']
        high = median(a['impact'] for a in coalesced) if coalesced else None
        assert high == row['high_impact_threshold']
        applicable = [(i, l) for i, l in enumerate(self.train_labels) if label_applies(l, row['country'])]
        result = dict(country=row['country'], variant=row['variant'], selection=row['selection'],
            impacts=[a['impact'] for a in coalesced],
            applicable_labels=[i for i, _ in applicable],
            references=[dict(impact=a['impact'], labels=[i for i, l in applicable if matches(a, l, self.proximity)])
                        for a in coalesced])
        write(cached, result)
        return result

    def detect(self, country, algorithm, parameters, annotated=False):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'parameters.json'
            write(path, parameters)
            command = [str(self.binary), 'detect', '--algorithm', algorithm.replace('_', '-'),
                '--input', str(self.root / f'inputs/{country}-evaluation.json'), '--parameters', str(path)]
            if algorithm in ('chebyshev', 'ensemble'):
                command += ['--reference-end', self.cutoff]
            if annotated:
                command.append('--include-annotated')
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            return json.loads(run.stdout)

    def members(self, row):
        country, variant, selection = (row[k] for k in ('country', 'variant', 'selection'))
        cached = self.cache / f'{country}-{variant}-{selection}-members.json'
        if cached.exists():
            return read(cached)
        folder = self.folder(country, variant)
        parameters = read(folder / f'{selection}-parameters.json')
        ensemble = self.detect(country, 'ensemble', parameters, annotated=True)
        saved = read(folder / f'{selection}-detection.json')
        assert ensemble['anomalies'] == saved['anomalies'], (country, variant, selection)
        run = read(folder / 'result.json')
        maximum = max(p['rate'] for p in read(self.root / f'inputs/{country}-evaluation.json'))
        output, events = [], []
        periods = [('training', run['first_date'], self.cutoff), ('test', self.first_test, self.config['test_end'])]
        for member in MEMBERS:
            p = member_parameters(parameters, member, run['training_maximum'], maximum)
            independent = self.detect(country, member, p)['anomalies']
            rejected = rejected_episodes(ensemble['annotated'], member)
            for period, first, last in periods:
                required = required_labels(self.labels, country, first, last, 0)
                applicable = [l for l in self.labels if first <= l['date'] <= last and label_applies(l, country)]
                saved_period = [a for a in saved['anomalies'] if first <= a['start'] <= last]
                ensemble_labels = coverage(saved_period, required, self.proximity)
                for mode, anomalies in [('standalone_shared_parameters', independent), ('rejected_single_vote', rejected)]:
                    selected = [a for a in anomalies if first <= a['start'] <= last]
                    covered = coverage(selected, required, self.proximity)
                    additional = [l for l in covered if l not in ensemble_labels]
                    output.append(dict(country=country, configuration=variant, selection=selection,
                        period=period, member=member, mode=mode, anomalies=len(selected),
                        dense_anomalies=sum(not a.get('sparse', False) for a in selected),
                        label_matched_anomalies=sum(any(matches(a, l, self.proximity) for l in applicable) for a in selected),
                        country_labels_total=len(required), country_labels_covered=len(covered),
                        country_labels_not_covered_by_ensemble=len(additional)))
                    for l in additional:
                        events.append(dict(analysis=mode, country=country, configuration=variant,
                            selection=selection, period=period, member=member, date=l['date'],
                            event=event_text(l), starts=';'.join(a['start'] for a in selected if matches(a,l,self.proximity))))
        result = dict(summary=output, events=events)
        write(cached, result)
        return result

    def sample(self, references, repeats):
        rows = self.context['countries']
        reference_map = {(r['country'], r['variant'], r['selection']): r for r in references}
        groups = defaultdict(list)
        for row in rows:
            ref = reference_map[row['country'], row['variant'], row['selection']]
            required = required_labels(self.labels, row['country'], self.first_test, self.config['test_end'], 0)
            anomalies = row['coalesced_anomalies']
            covered_by = [{i for i,l in enumerate(required) if matches(a,l,self.proximity)} for a in anomalies]
            groups[row['variant'], row['selection']].append((row, ref, covered_by))
        samples = []
        size = len(self.train_labels)
        for iteration in range(repeats):
            rng = random.Random(20261023 + iteration)
            order = list(range(size)); rng.shuffle(order)
            # Country-specific draws isolate the zero-to-one transition.
            applicable_by_country = {r['country']: r['applicable_labels'] for r in references}
            one = {c: rng.choice(ids) if ids else None
                   for c, ids in sorted(applicable_by_country.items())}
            draws = {c: rng.random() for c in sorted(applicable_by_country)}
            for mode, fraction in [('fraction', f) for f in FRACTIONS] + [
                    ('one_per_country', None), ('one_explained_per_country', None)]:
                selected = set(order[:math.floor(fraction*size)]) if mode == 'fraction' else set()
                for (variant, selection), items in groups.items():
                    removed = retained_labels = calibrated = retained = sparse_removed = 0
                    for row, ref, covered_by in items:
                        if mode == 'fraction':
                            chosen = selected
                        elif mode == 'one_per_country':
                            chosen = {one[row['country']]} if one[row['country']] is not None else set()
                        else:
                            explained = sorted({i for r in ref['references'] for i in r['labels']})
                            chosen = {explained[int(draws[row['country']]*len(explained))]} if explained else set()
                        threshold = sampling_threshold(ref['references'], chosen)
                        calibrated += threshold is not None
                        covered = set()
                        for a, supported in zip(row['coalesced_anomalies'], covered_by, strict=True):
                            if kept(a['impact'], threshold, row['high_impact_threshold']):
                                retained += 1; covered.update(supported)
                            else:
                                removed += 1; sparse_removed += a.get('sparse', False)
                        retained_labels += len(covered)
                    if fraction == 1:
                        expected = self.context['summary'][f'{variant}/{selection}']
                        assert retained == expected['retained'] and retained_labels == expected['retained_covered']
                    if fraction == 0:
                        assert removed == 0 and calibrated == 0
                    samples.append(dict(iteration=iteration, mode=mode, fraction=fraction,
                        configuration=variant, selection=selection, filtered=removed, sparse_filtered=sparse_removed,
                        retained=retained, test_labels_retained=retained_labels, calibrated_countries=calibrated))
        grouped = defaultdict(list)
        for row in samples:
            grouped[row['configuration'],row['selection'],row['mode'],row['fraction']].append(row)
        summary = []
        for (v,s,m,f), batch in grouped.items():
            r = dict(configuration=v,selection=s,mode=m,fraction=f,repeats=repeats)
            for metric in ('filtered','sparse_filtered','retained','test_labels_retained','calibrated_countries'):
                values = [x[metric] for x in batch]
                r[metric] = dict(mean=mean(values), p025=quantile(values,.025), p975=quantile(values,.975))
            summary.append(r)
        return dict(seed=20261023, repeats=repeats, training_label_pool=size,
                    summary=summary, iterations=samples)

    def notifications(self, references):
        reference_map = {(r['country'],r['variant'],r['selection']): r['impacts'] for r in references}
        summaries, details = [], []
        for key, grouping in self.context['available_grouping'].items():
            variant, selection = key.split('/')
            g = grouping['5']; counts = Counter(); levels = Counter()
            entries = [(a['start'], [a], annotated['labels']) for a, annotated in zip(
                g['individual_notifications'],g['label_matching']['individuals'],strict=True)]
            entries += [(a['date'],a['anomalies'],[l for m in annotated['members'] for l in m['labels']])
                        for a,annotated in zip(g['grouped_notifications'],g['label_matching']['groups'],strict=True)]
            for when, anomalies, labels in entries:
                types = {label_kind(l) for l in labels}
                category = ('restriction_related' if 'restriction_related' in types else
                            'other_explained' if types else 'unexplained')
                rank = {'unavailable':-1,'low':0,'medium':1,'high':2,'very_high':3}
                level = max((severity(a['impact'],reference_map[a['country'],variant,selection]) for a in anomalies),key=rank.get)
                counts[category] += 1; levels[level] += 1
                details.append(dict(analysis='notification',configuration=variant,selection=selection,
                    country=';'.join(a['country'] for a in anomalies),date=when,
                    category=category,severity=level,event='; '.join(dict.fromkeys(event_text(l) for l in labels))))
            assert sum(counts.values()) == g['label_matching']['total']
            assert counts['restriction_related'] + counts['other_explained'] == g['label_matching']['matched']
            summaries.append(dict(configuration=variant,selection=selection,categories=dict(counts),severity=dict(levels)))
        return summaries, details

    def comparisons(self, references):
        reference_map = {(r['country'],r['variant'],r['selection']): r['impacts'] for r in references}
        result, events = [], []
        for variant in ('original','labels','ensemble','labels_ensemble'):
            by_country = defaultdict(dict)
            for row in self.context['countries']:
                if row['variant'] == variant:by_country[row['country']][row['selection']]=row
            for country, selected in by_country.items():
                required = required_labels(self.labels,country,self.first_test,self.config['test_end'],0)
                for stage in ('raw_anomalies','retained_anomalies'):
                    k=coverage(selected['knee'][stage],required,self.proximity)
                    l=coverage(selected['leftmost'][stage],required,self.proximity)
                    for label in required:
                        if (label in k) != (label in l):
                            events.append(dict(analysis='selection_difference',configuration=variant,country=country,
                                stage=stage,date=label['date'],event=event_text(label),
                                selection='knee_only' if label in k else 'leftmost_only'))
            for selection in ('knee','leftmost'):
                selected=[r for r in self.context['countries'] if r['variant']==variant and r['selection']==selection]
                training_count=0
                for r in selected:
                    anomalies=read(self.folder(r['country'],variant)/f'{selection}-detection.json')['anomalies']
                    training_count+=sum(a['start']<=self.cutoff for a in anomalies)
                summary=self.context['summary'][f'{variant}/{selection}']
                result.append(dict(configuration=variant,selection=selection,training_anomalies=training_count,
                    training_labels_covered=summary['training_labels_covered'],training_labels_total=summary['training_labels_total'],
                    test_anomalies=summary['raw'],test_labels_covered=summary['raw_covered'],
                    notifications=summary['notifications']['5'],retained_labels=summary['retained_covered']))
        for row in self.context['countries']:
            for label in row['lost_labels']:
                matched=[a for a in row['coalesced_anomalies'] if matches(a,label,self.proximity)]
                reference=reference_map[row['country'],row['variant'],row['selection']]
                events.append(dict(analysis='filtered_test_label',configuration=row['variant'],selection=row['selection'],
                    country=row['country'],date=label['date'],event=event_text(label),
                    starts=';'.join(a['start'] for a in matched),
                    impact=';'.join(str(a['impact']) for a in matched),
                    severity=';'.join(severity(a['impact'],reference) for a in matched)))
        return result, events


def aggregate_members(results):
    totals = defaultdict(Counter)
    fields=('anomalies','dense_anomalies','label_matched_anomalies','country_labels_total','country_labels_covered','country_labels_not_covered_by_ensemble')
    for result in results:
        for row in result['summary']:
            totals[tuple(row[k] for k in ('configuration','selection','period','member','mode'))].update({k:row[k] for k in fields})
    return [dict(zip(('configuration','selection','period','member','mode'), key),**values,
                 label_match_rate=values['label_matched_anomalies']/values['anomalies'] if values['anomalies'] else None)
            for key,values in sorted(totals.items())]


def markdown(result):
    indexed={(r['configuration'],r['selection']):r for r in result['comparisons']}
    k,l=indexed['labels_ensemble','knee'],indexed['labels_ensemble','leftmost']
    sample={r['fraction']:r for r in result['sampling']['summary'] if r['configuration']=='labels_ensemble'
            and r['selection']=='knee' and r['mode']=='fraction'}
    one=next(r for r in result['sampling']['summary'] if r['configuration']=='labels_ensemble'
             and r['selection']=='knee' and r['mode']=='one_explained_per_country')
    lines=['# Results analysis', '',
        '71 countries; training through 2024, testing January 2025–August 2026. '
        'Analyses use the saved parameter sets and final labels. Detailed results for all configurations '
        'and sampling repetitions are in `analysis.json`; event-level records are in `event-review.csv`.', '',
        '## Main observations', '',
        f"- Constrained ensemble: knee produces {k['notifications']} notifications retaining {k['retained_labels']} country labels; leftmost produces {l['notifications']} retaining {l['retained_labels']}. Both preserve all {k['training_labels_total']} training labels.",
        '- Label sampling for constrained ensemble/knee: using 25%, 50%, 75%, and 100% of training labels filters '
        + ', '.join(f"{sample[f]['filtered']['mean']:.1f}" for f in (.25,.5,.75,1))
        + ' test anomalies on average, retaining '
        + ', '.join(f"{sample[f]['test_labels_retained']['mean']:.1f}" for f in (.25,.5,.75,1))
        + ' country labels. No training labels means no impact filtering.',
        f"- One explained training label per calibrated country filters {one['filtered']['mean']:.1f} anomalies and retains {one['test_labels_retained']['mean']:.1f} test country labels on average; the complete label set filters {sample[1]['filtered']['mean']:.0f} and retains {sample[1]['test_labels_retained']['mean']:.0f}.", '',
        '## Label constraints and selection', '',
        '| Configuration | Selection | Training anomalies | Train labels | Test anomalies | Test labels | Notifications | Retained labels |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['comparisons']:
        lines.append(f"| {r['configuration']} | {r['selection']} | {r['training_anomalies']} | {r['training_labels_covered']}/{r['training_labels_total']} | {r['test_anomalies']} | {r['test_labels_covered']}/61 | {r['notifications']} | {r['retained_labels']}/61 |")
    extra=[r for r in result.get('event_differences',[]) if r['analysis']=='selection_difference'
           and r['configuration']=='labels_ensemble' and r['stage']=='retained_anomalies']
    if extra:
        lines += ['', 'Constrained ensemble selection differences after contextualization:', '']
        for r in extra:lines.append(f"- {r['selection']}: {r['country']} {r['date']} — {r['event']}")
    lines += ['', '## Individual detectors', '',
        'Each member runs with the selected label-constrained ensemble parameters, its own evolving '
        'window, and the same training-calibrated absolute Croston threshold. Counts include sparse '
        'detections. Matched anomalies have a country or global label; coverage counts country labels.', '',
        '| Configuration | Selection | Period | Member | Anomalies (dense) | Matched anomalies (rate) | Country labels |',
        '|---|---|---|---|---:|---:|---:|']
    for r in result['members']:
        if r['mode']=='standalone_shared_parameters' and r['configuration']=='labels_ensemble':
            rate=f"{r['label_match_rate']:.1%}" if r['label_match_rate'] is not None else 'n/a'
            lines.append(f"| {r['configuration']} | {r['selection']} | {r['period']} | {r['member']} | {r['anomalies']} ({r['dense_anomalies']}) | {r['label_matched_anomalies']} ({rate}) | {r['country_labels_covered']}/{r['country_labels_total']} |")
    lines += ['', '## Majority-vote rejections', '',
        'In the ensemble’s shared state, consecutive days with exactly one positive member vote and no '
        'retained ensemble anomaly form a rejected episode. Matching uses episode start dates, with '
        '±3 days for country labels and ±1 day for global labels. “Missed labels” counts country labels matched by these episodes but absent from '
        'the ensemble’s detections; labels supported by multiple members appear in multiple rows.', '',
        '| Configuration | Selection | Period | Member | Rejected episodes | Label-matched episodes | Missed labels |',
        '|---|---|---|---|---:|---:|---:|']
    for r in result['members']:
        if r['mode']=='rejected_single_vote' and r['configuration']=='labels_ensemble' and r['period']=='test':
            lines.append(f"| {r['configuration']} | {r['selection']} | {r['period']} | {r['member']} | {r['anomalies']} | {r['label_matched_anomalies']} | {r['country_labels_not_covered_by_ensemble']} |")
    lines += ['', '## Notifications', '',
        'Restriction-related means a matched label describes blocking, throttling, shutdown/restoration, '
        'or services in the supplied restriction lists. Other explained events include global events, '
        'VPN discussion, policy reactions, and other non-blocking causes. A notification with both types '
        'is counted as restriction-related. '
        'Severity uses each country/configuration’s coalesced training-impact quartiles; ties go to '
        'the higher band. A cross-country group takes its highest member severity.', '',
        '| Configuration | Selection | Restriction-related | Other explained | Unexplained | Low | Medium | High | Very high | Unavailable |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['notifications']:
        c,s=r['categories'],r['severity']
        lines.append('| '+ ' | '.join(map(str,[r['configuration'],r['selection'],*[c.get(k,0) for k in ('restriction_related','other_explained','unexplained')],*[s.get(k,0) for k in ('low','medium','high','very_high','unavailable')]]))+' |')
    lost=[r for r in result.get('event_differences',[]) if r['analysis']=='filtered_test_label'
          and r['configuration']=='labels_ensemble' and r['selection']=='knee']
    if lost:
        lines += ['', 'Country labels removed by impact filtering in the constrained ensemble/knee configuration:', '']
        for r in lost:lines.append(f"- {r['country']} {r['date']} — {r['event']} ({r['severity']} severity).")
    lines += ['', '## Training-label subsampling', '',
        '100 repetitions, seed 20261023 + repetition. At each repetition, shuffle the full training-label '
        'pool once and take nested subsets at 0%, 1%, 5%, 10%, 25%, 50%, 75%, and 100% (floor of pool size × fraction). '
        'A global label enters or leaves all countries together. Sampled labels calibrate impact thresholds; '
        'detection parameters, training-impact severity boundaries, and test labels stay fixed. '
        'The additional one-label-per-country condition draws one applicable training label uniformly '
        'for each country. A label can be unmatched by that configuration, leaving impact filtering disabled. '
        'The one-explained-label condition draws from labels that match a training anomaly for that '
        'country/configuration; countries with no such label remain unfiltered. '
        'Values are means; brackets show empirical 2.5th–97.5th percentiles over label draws.', '',
        'Results below use the label-constrained ensemble with knee selection.', '',
        '| Configuration | Selection | Labels available | Filtered anomalies | Retained test labels | Calibrated countries |',
        '|---|---|---|---:|---:|---:|']
    for r in result['sampling']['summary']:
        if r['configuration']!='labels_ensemble' or r['selection']!='knee':continue
        display = {'one_per_country':'one/country','one_explained_per_country':'one explained/country'}.get(r['mode'])
        if display is None:display=f"{100*r['fraction']:g}%"
        def value(k):
            v=r[k];return f"{v['mean']:.1f} [{v['p025']:.1f}, {v['p975']:.1f}]"
        lines.append(f"| {r['configuration']} | {r['selection']} | {display} | {value('filtered')} | {value('test_labels_retained')} | {value('calibrated_countries')} |")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--contextualization',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args()
    if args.workers<1 or args.repeats<2:parser.error('workers >= 1 and repeats >= 2 required')
    analysis=Analysis(args.results.resolve(),args.contextualization.resolve(),args.cache.resolve())
    signature = {str(p.relative_to(analysis.root)): [p.stat().st_size,p.stat().st_mtime_ns]
                 for p in sorted(analysis.root.rglob('*')) if p.is_file()
                 and (p.parent.name=='inputs' or p.name.endswith('-detection.json')
                      or p.name.endswith('-parameters.json') or p.name=='experiment.json')}
    signature['contextualization'] = [str(args.contextualization.resolve()),
                                      args.contextualization.stat().st_mtime_ns]
    cache_manifest=analysis.cache/'inputs.json'
    if cache_manifest.exists() and read(cache_manifest)!=signature:
        raise ValueError('Analysis inputs changed; use a fresh cache directory.')
    write(cache_manifest,signature)
    with ThreadPoolExecutor(args.workers) as pool:
        references=[]
        for n,r in enumerate(pool.map(analysis.calibrate,analysis.context['countries']),1):
            references.append(r)
            if n%100==0:print(f'Calibrated saved detections: {n}/{len(analysis.context["countries"])}',flush=True)
        tasks=[r for r in analysis.context['countries'] if 'ensemble' in r['variant']]
        member_results=[]
        for n,r in enumerate(pool.map(analysis.members,tasks),1):
            member_results.append(r)
            if n%20==0:print(f'Member detection analysis: {n}/{len(tasks)}',flush=True)
    print('Sampling training labels for impact calibration',flush=True)
    sampling=analysis.sample(references,args.repeats)
    notifications,events=analysis.notifications(references)
    comparisons,comparison_events=analysis.comparisons(references)
    events+=comparison_events+[e for r in member_results for e in r['events']]
    with tempfile.TemporaryDirectory() as temporary:
        accepted_member_analysis(analysis.root,Path(temporary))
        with (Path(temporary)/'member-contributions.csv').open() as stream:
            accepted_counts=list(csv.DictReader(stream))
        with (Path(temporary)/'label-contributions.csv').open() as stream:
            accepted_labels=list(csv.DictReader(stream))
    result=dict(method='Fixed-parameter detection and contextualization analysis',
        comparisons=comparisons,members=aggregate_members(member_results),
        accepted_member_votes=accepted_counts, accepted_label_votes=accepted_labels,
        notifications=notifications,sampling=sampling,event_differences=comparison_events)
    write(analysis.root/'analysis.json',result)
    (analysis.root/'analysis.md').write_text(markdown(result))
    fields=list(dict.fromkeys(k for r in events for k in r))
    with (analysis.root/'event-review.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(events)
    print('Wrote analysis.md, analysis.json, and event-review.csv',flush=True)


if __name__=='__main__':main()
