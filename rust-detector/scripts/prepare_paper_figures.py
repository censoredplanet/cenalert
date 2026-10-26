"""Plot detection results, label sensitivity, and notification severity."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from analyze_paper_results import event_text, label_kind
from batch_ensemble_tuning import label_applies
from review_contextualization import grouping_key, matches, read, write
from paper_plot_style import apply_paper_style

CHOSEN = 'labels_ensemble/knee'
REVIEW_HEADING = '## Same-day country groups'
VARIANTS = ('original', 'labels', 'ensemble', 'labels_ensemble', 'fixed_3sigma')
NAMES = ('Single detector', 'Single + labels', 'Ensemble', 'Ensemble + labels', r'Fixed $3\sigma$')


def same_day_groups(records, labels, proximity):
    """Count distinct countries with same-day retained anomalies and restriction labels."""
    days = defaultdict(list)
    for record in records:
        for anomaly in record['retained_anomalies']:
            days[anomaly['start']].append(dict(anomaly, country=record['country']))
    rows = []
    for day, anomalies in sorted(days.items()):
        countries = sorted({a['country'] for a in anomalies})
        exact, nearby = set(), set()
        for anomaly in anomalies:
            for label in labels:
                if label['scope'] != 'country' or not label_applies(label, anomaly['country']):
                    continue
                on_day = label['date'] == day
                in_window = matches(anomaly, label, proximity)
                if not (on_day or in_window) or label_kind(label) != 'restriction_related':
                    continue
                item = (anomaly['country'], label['date'], event_text(label))
                if on_day:
                    exact.add(item)
                if in_window:
                    nearby.add(item)
        describe = lambda items: '; '.join(f'{c} {d}: {e}' for c, d, e in sorted(items))
        rows.append(dict(date=day, country_count=len(countries), countries=';'.join(countries),
            same_day_restriction_countries=';'.join(sorted({x[0] for x in exact})),
            same_day_restriction_events=describe(exact),
            window_matched_restriction_countries=';'.join(sorted({x[0] for x in nearby})),
            window_matched_restriction_events=describe(nearby)))
    return rows


def group_summary(rows):
    return dict(multi_country_dates=sum(r['country_count'] >= 2 for r in rows),
        largest_same_day_restriction_group=max((r['country_count'] for r in rows
            if r['same_day_restriction_events']), default=0),
        largest_window_matched_restriction_group=max((r['country_count'] for r in rows
            if r['window_matched_restriction_events']), default=0),
        groups_at_least_five=sum(r['country_count'] >= 5 for r in rows),
        groups_at_least_five_with_same_day_restriction=sum(r['country_count'] >= 5 and
            bool(r['same_day_restriction_events']) for r in rows),
        groups_at_least_five_with_window_match=sum(r['country_count'] >= 5 and
            bool(r['window_matched_restriction_events']) for r in rows))


def matched_notifications(context, events):
    g = context['available_grouping'][CHOSEN][grouping_key(context)]
    details = {(r['date'], r['country']): r for r in events
               if r['analysis'] == 'notification' and
               f"{r['configuration']}/{r['selection']}" == CHOSEN}
    entries = [(a['start'], [a], [m]) for a, m in zip(g['individual_notifications'],
                g['label_matching']['individuals'], strict=True)]
    entries += [(a['date'], a['anomalies'], m['members']) for a, m in zip(
                g['grouped_notifications'], g['label_matching']['groups'], strict=True)]
    rows = []
    for day, anomalies, members in entries:
        labels = {(m['country'], l['scope'], l['date'], event_text(l), label_kind(l))
                  for m in members for l in m['labels']}
        if not labels:
            continue
        countries = ';'.join(a['country'] for a in anomalies)
        detail = details[day, countries]
        category = 'restriction_related' if any(l[-1] == 'restriction_related' for l in labels) else 'other_explained'
        assert detail['category'] == category
        rows.append(dict(notification_date=day, countries=countries,
            notification_type='group' if len(anomalies) > 1 else 'individual',
            category=category, severity=detail['severity'], event=detail['event'],
            matched_labels='; '.join(f'{c} / {s} / {d} / {e} / {kind}'
                                    for c, s, d, e, kind in sorted(labels))))
    assert len(rows) == g['label_matching']['matched']
    return sorted(rows, key=lambda r: (r['notification_date'], r['countries']))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No records for {path.name}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, directory, name):
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_detection(context, directory):
    rows = [context['available_summary'][f"{v}/{'fixed' if v == 'fixed_3sigma' else 'knee'}"] for v in VARIANTS]
    assert len({r['countries'] for r in rows}) == len({r['labels_total'] for r in rows}) == 1
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), layout='constrained')
    y = list(range(len(rows)))
    for ax, before, after, xlabel, before_name, after_name in (
        (axes[0], 'raw', None, 'Test detections / notifications', 'Raw anomalies', 'Final notifications'),
        (axes[1], 'raw_covered', 'retained_covered', f"Country test labels covered (of {rows[0]['labels_total']})", 'Before contextualization', 'After contextualization')):
        b = [r[before] for r in rows]
        a = [r[after] if after else r['notifications'][grouping_key(context)] for r in rows]
        ax.barh([i-.17 for i in y], b, height=.30, color='#9ebbd2', label=before_name)
        ax.barh([i+.17 for i in y], a, height=.30, color='#245a81', label=after_name)
        for i, x, z in zip(y, b, a):
            ax.text(x+max(b)*.015, i-.17, str(x), va='center', fontsize=8)
            ax.text(z+max(b)*.015, i+.17, str(z), va='center', fontsize=8)
        ax.set_yticks(y, NAMES if ax is axes[0] else ['']*len(y))
        ax.invert_yaxis()
        ax.set_xlim(0, max(b)*1.15 if after is None else rows[0]['labels_total'])
        ax.set_xlabel(xlabel)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), fontsize=7)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    save(fig, directory, 'detection-contextualization')


def plot_sampling(analysis, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), layout='constrained')
    rows = sorted((r for r in analysis['sampling']['summary'] if r['mode']=='fraction'
        and f"{r['configuration']}/{r['selection']}" == CHOSEN), key=lambda r:r['fraction'])
    for ax, field, title, color in zip(axes, ('filtered', 'test_labels_retained'),
        ('Test anomalies removed by impact', 'Country test labels retained'), ('#245a81', '#ad621c')):
        x = [r['fraction']*100 for r in rows]
        ax.plot(x, [r[field]['mean'] for r in rows], 'o-', color=color, ms=3)
        ax.fill_between(x, [r[field]['p025'] for r in rows], [r[field]['p975'] for r in rows], color=color, alpha=.18)
        ax.set(xlabel='Training labels sampled (%)', ylabel=title, xlim=(0, 100))
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(bottom=0)
        ax.grid()
    save(fig, directory, 'label-sampling')


def plot_severity(analysis, directory):
    rows = analysis['alert_quality']['severity_by_category']
    rows = [next(r for r in rows if r['category']==k)
            for k in ('restriction_related', 'other_explained', 'unexplained')]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')
    bottoms = [0.0]*3
    for level, color in zip(('low','medium','high','very_high'), ('#d8e5ee','#9ebbd2','#4e88b2','#19496e')):
        values = [r[level]/r['total'] for r in rows]
        ax.bar(range(3), values, bottom=bottoms, color=color, label=level.replace('_',' ').title(), width=.62)
        for i, value in enumerate(values):
            if value:
                ax.text(i, bottoms[i]+value/2, str(rows[i][level]), ha='center', va='center',
                        color='white' if level in ('high','very_high') else '#222222', fontsize=8)
        bottoms = [b+v for b,v in zip(bottoms,values)]
    ax.set_xticks(range(3), [f'Restriction-\nrelated\n(n={rows[0]["total"]})',
        f'Other\nexplained\n(n={rows[1]["total"]})', f'Unexplained\n(n={rows[2]["total"]})'])
    ax.set_ylabel('Share of notifications')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1)
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.01), fontsize=7)
    save(fig, directory, 'notification-severity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    context, analysis = read(root/'contextualization.json'), read(root/'analysis.json')
    labels = read(root/'inputs/labels.json')
    labels = [l for l in labels if context['train_end'] < l['date'] <= context['test_end']]
    with (root/'event-review.csv').open(newline='') as stream:
        events = list(csv.DictReader(stream))
    matched = matched_notifications(context, events)
    write_csv(root/'matched-notifications.csv', matched)
    summaries, primary, large_restriction_groups = {}, [], []
    for key in context['available_summary']:
        records = [r for r in context['countries'] if f"{r['variant']}/{r['selection']}" == key]
        rows = same_day_groups(records, labels, context['country_proximity_days'])
        summaries[key] = group_summary(rows)
        large_restriction_groups.extend(dict(configuration=key, **r) for r in rows
            if r['country_count'] >= 5 and (r['same_day_restriction_events'] or
                                           r['window_matched_restriction_events']))
        if key == CHOSEN:
            primary = [r for r in rows if r['country_count'] >= 2]
    write_csv(root/'same-day-groups.csv', primary)
    analysis['same_day_review'] = dict(stage='after impact filtering, before country grouping',
        period='test', summary=summaries, groups=primary,
        large_restriction_groups=large_restriction_groups)
    write(root/'analysis.json', analysis)
    lines = [REVIEW_HEADING, '',
        'Counts below use distinct countries with retained anomalies starting on the same date. '
        'Exact-day evidence uses the recorded country-label date; window evidence uses the existing '
        'country matching window and coalesced component starts. Event categories follow the notification review.', '',
        '| Configuration / selection | Largest group with an exact-day restriction label | Largest group with a window-matched restriction label |',
        '|---|---:|---:|']
    for key, r in summaries.items():
        lines.append(f"| {key} | {r['largest_same_day_restriction_group']} | {r['largest_window_matched_restriction_group']} |")
    lines += ['', 'Constrained ensemble / knee, groups containing restriction-related matches:', '']
    for r in primary:
        if r['same_day_restriction_events'] or r['window_matched_restriction_events']:
            lines.append(f"- {r['date']}: {r['country_count']} countries ({r['countries']}). "
                f"Exact day: {r['same_day_restriction_events'] or 'none'}. "
                f"Matching window: {r['window_matched_restriction_events'] or 'none'}.")
    s = summaries[CHOSEN]
    lines += ['', f"Among the {s['groups_at_least_five']} groups with at least five countries, "
        f"{s['groups_at_least_five_with_same_day_restriction']} have an exact-day restriction label and "
        f"{s['groups_at_least_five_with_window_match']} have a window-matched restriction label.", '']
    for r in large_restriction_groups:
        lines.append(f"- {r['configuration']}, {r['date']}: {r['country_count']} countries "
            f"({r['countries']}). Exact day: {r['same_day_restriction_events'] or 'none'}. "
            f"Matching window: {r['window_matched_restriction_events'] or 'none'}.")
    lines += ['',
        '### Figure captions', '',
        f"**detection-contextualization.pdf:** Test detections and country-label coverage across {context['available_summary'][CHOSEN]['countries']} countries. "
        'Tuned configurations use knee/fallback selection. All configurations use training labels during contextualization.', '',
        '**label-sampling.pdf:** Effect of training-label availability on impact filtering for the label-constrained ensemble '
        f"with knee/fallback selection. Lines show means over {analysis['sampling']['repeats']} random subsets; shading shows the 2.5th-97.5th percentiles. "
        'Detection parameters and severity boundaries remain fixed. '
        f"Country-label coverage uses all {context['available_summary'][CHOSEN]['labels_total']} test labels.", '',
        f"**notification-severity.pdf:** Severity of the {context['available_summary'][CHOSEN]['notifications'][grouping_key(context)]} final notifications from the label-constrained ensemble "
        'with knee/fallback selection. Bar heights show within-category proportions; segment numbers show counts. '
        'Severity boundaries use each country’s coalesced training-impact distribution.', '']
    path = root/'analysis.md'
    text = path.read_text().split(REVIEW_HEADING)[0].rstrip()
    path.write_text(text+'\n\n'+'\n'.join(lines))
    directory = root/'figures'
    directory.mkdir(exist_ok=True)
    apply_paper_style()
    plot_detection(context, directory)
    plot_sampling(analysis, directory)
    plot_severity(analysis, directory)
    print('Matched notifications:', dict(Counter(r['category'] for r in matched)))
    print('Same-day groups:', summaries[CHOSEN])
    print('Wrote three figures, matched-notifications.csv, same-day-groups.csv, and updated analysis.')


if __name__ == '__main__':
    main()
