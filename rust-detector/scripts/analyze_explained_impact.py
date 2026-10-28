"""Compare impact of restriction-related, other explained, and unmatched detections."""
import argparse
from collections import defaultdict
from statistics import mean, median

from pathlib import Path
from analyze_paper_results import label_kind
from batch_ensemble_tuning import label_applies
from review_contextualization import matches, read, write
from prepare_paper_figures import write_csv

CATEGORIES = ('restriction_related', 'other_explained', 'unexplained')


def replace_section(text, heading, body):
    before, found, after = text.partition(heading)
    end = after.find('\n## ') if found else -1
    suffix = after[end:] if end >= 0 else ''
    return before.rstrip() + '\n\n' + body.rstrip() + '\n' + suffix


def category_for(anomaly, labels, proximity):
    kinds = {label_kind(l) for l in labels if matches(anomaly, l, proximity)}
    return ('restriction_related' if 'restriction_related' in kinds else
            'other_explained' if kinds else 'unexplained')


def summarize(records, total):
    result = []
    for category in CATEGORIES:
        values = [r['impact'] for r in records if r['category'] == category]
        result.append(dict(category=category, anomalies=len(values),
            mean_impact=mean(values) if values else None, median_impact=median(values) if values else None,
            cumulative_impact=sum(values), impact_share=sum(values)/total if total else None))
    return result


def analyze(context, labels):
    labels = [l for l in labels if context['train_end'] < l['date'] <= context['test_end']]
    groups = defaultdict(list)
    for row in context['countries']:
        applicable = [l for l in labels if label_applies(l, row['country'])]
        for stage in ('raw_anomalies', 'retained_anomalies'):
            for a in row[stage]:
                groups[row['variant'], row['selection'], stage].append(dict(country=row['country'],
                    start=a['start'], impact=a['impact'], category=category_for(
                        a, applicable, context['country_proximity_days'])))
    summaries, by_country = [], []
    for (variant, selection, stage), records in groups.items():
        total = sum(r['impact'] for r in records)
        countries = defaultdict(list)
        for r in records:
            countries[r['country']].append(r)
        shares = [sum(r['impact'] for r in rs if r['category'] != 'unexplained')/
                  sum(r['impact'] for r in rs) for rs in countries.values() if sum(r['impact'] for r in rs)>0]
        summary = dict(configuration=variant, selection=selection, stage=stage,
            anomalies=len(records), total_impact=total, categories=summarize(records, total),
            mean_country_explained_share=mean(shares) if shares else None,
            countries_with_positive_impact=len(shares))
        summaries.append(summary)
        if (variant, selection, stage) == ('labels_ensemble', 'knee', 'retained_anomalies'):
            for country, rows in sorted(countries.items()):
                by_country.extend(dict(country=country, **r) for r in summarize(rows, sum(x['impact'] for x in rows)))
    return dict(unit='Anomaly intervals; retained intervals before cross-country notification grouping',
        classification='Restriction-related takes precedence when both country and global labels match',
        country_average='Equal weight for each country with positive cumulative impact at the selected stage',
        summaries=summaries, countries=by_country)


def country_impact_rows(result, limit=12):
    """Rank countries by retained impact and aggregate the remaining countries."""
    countries = defaultdict(dict)
    for row in result['countries']:
        countries[row['country']][row['category']] = row
    ordered = sorted(countries, key=lambda c: (
        -sum(r['cumulative_impact'] for r in countries[c].values()), c))
    rows = [(c, countries[c]) for c in ordered[:limit]]
    remainder = ordered[limit:]
    if remainder:
        rows.append((f'Remaining {len(remainder)} countries', {
            category: {field: sum(countries[c][category][field] for c in remainder)
                       for field in ('cumulative_impact', 'anomalies')}
            for category in CATEGORIES}))
    return rows


def plot(result, output, country_names=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_plot_style import apply_paper_style
    apply_paper_style()
    country_names = dict(country_names or {})
    country_names.update(CD='DR Congo', IR='Iran', TZ='Tanzania', VN='Vietnam')
    rows = country_impact_rows(result)
    fig, ax = plt.subplots(figsize=(6.8, 3.65))
    names = ['Restriction-related', 'Other explained', 'Unmatched']
    colors = ['crimson', 'purple', 'darkslategray']
    positions = list(range(len(rows)))
    has_remainder = bool(rows and rows[-1][0].startswith('Remaining '))
    if has_remainder:
        positions[-1] += .45
    left = [0.0] * len(rows)
    for category, name, color in zip(CATEGORIES, names, colors):
        values = [r[category]['cumulative_impact'] for _, r in rows]
        ax.barh(positions, values, left=left, height=.6, color=color, label=name)
        left = [a+b for a, b in zip(left, values)]
    ax.set_yticks(positions, [country_names.get(c, c) for c, _ in rows])
    ax.set_ylim(positions[-1]+.65, -.65)
    ax.set_xlim(0, max(left)*1.02)
    ax.set_xlabel('Cumulative Impact Factor')
    ax.grid(axis='both', linestyle=':', linewidth=.5, alpha=.6)
    ax.tick_params(axis='y', length=0)
    if has_remainder:
        ax.axhline(positions[-1]-.75, color='.65', linewidth=.5)

    # Counts accompany the impact bars, as in the original CenAlert Figure 6.
    # Fixed columns keep short bars and large category counts equally legible.
    count_columns = [1.045, 1.115, 1.185]
    for x, short_name, color in zip(count_columns, ['R', 'O', 'U'], colors):
        ax.text(x, 1.015, short_name, color=color, ha='center', va='bottom',
                transform=ax.transAxes, fontsize=8, fontweight='bold')
    ax.text(1.115, 1.075, 'Anomalies', ha='center', transform=ax.transAxes, fontsize=8)
    for y, (_, row) in zip(positions, rows):
        for x, category, color in zip(count_columns, CATEGORIES, colors):
            ax.text(x, y, str(row[category]['anomalies']), color=color,
                    ha='center', va='center', transform=ax.get_yaxis_transform(), fontsize=8)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.14), ncol=3,
              frameon=False, borderaxespad=0, columnspacing=1.2, handlelength=1.2)
    fig.subplots_adjust(left=.19, right=.84, bottom=.14, top=.86)
    fig.savefig(output,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    labels = read(root/'inputs/labels.json')
    result = analyze(read(root/'contextualization.json'), labels)
    analysis = read(root/'analysis.json')
    analysis['explained_impact'] = result
    write(root/'analysis.json', analysis)
    write_csv(root/'impact-summary.csv', result['countries'])
    country_names = {l['country_code']: l.get('metadata', {}).get('country', l['country_code'])
                     for l in labels if l.get('country_code')}
    plot(result,root/'figures/explained-impact.pdf', country_names)
    chosen = next(r for r in result['summaries'] if (r['configuration'],r['selection'],r['stage']) ==
                  ('labels_ensemble','knee','retained_anomalies'))
    heading = '## Explained impact'
    text = (root/'analysis.md').read_text()
    lines = [heading, '', 'Impact statistics use retained anomaly intervals before country grouping.', '',
             '| Category | Intervals | Mean IF | Median IF | Cumulative IF | Share |',
             '|---|---:|---:|---:|---:|---:|']
    for r in chosen['categories']:
        lines.append(f"| {r['category']} | {r['anomalies']} | {r['mean_impact']:.2f} | {r['median_impact']:.2f} | "
                     f"{r['cumulative_impact']:.2f} | {r['impact_share']:.1%} |")
    lines += ['', f"The mean within-country explained-impact share is {chosen['mean_country_explained_share']:.1%} "
              f"across {chosen['countries_with_positive_impact']} countries with retained positive impact.", '']
    (root/'analysis.md').write_text(replace_section(text, heading, '\n'.join(lines)))
    print(chosen)


if __name__ == '__main__':
    main()
