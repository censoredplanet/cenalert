"""Plot saved held-out detection cases in the style of CenAlert Figure 4."""
import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from analyze_paper_results import Analysis
from review_contextualization import matches, read, write
from paper_plot_style import apply_paper_style

CASES = (
    ('RU', '2025-06-09', '2025-05-28', '2025-06-23', 'Russia: Cloudflare throttling'),
    ('TM', '2025-07-21', '2025-07-10', '2025-07-28', 'Turkmenistan: VPN blocking'),
)


def replay(analysis, country, variant):
    folder = analysis.folder(country, variant)
    saved = read(folder/'knee-detection.json')
    cached = analysis.cache/f'{country}-{variant}-case.json'
    if cached.exists():
        result = read(cached)
    else:
        result = analysis.detect(country, 'ensemble' if variant == 'labels_ensemble' else 'chebyshev',
                                 read(folder/'knee-parameters.json'), annotated=True)
        write(cached, result)
    assert result['anomalies'] == saved['anomalies'], (country, variant)
    return result


def plot(analysis, destination):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.35))
    evidence = []
    for ax, (country, when, first, last, title) in zip(axes, CASES):
        single, ensemble = (replay(analysis, country, v) for v in ('labels', 'labels_ensemble'))
        label = dict(scope='country', date=when)
        detected = [a for a in ensemble['anomalies'] if matches(a, label, analysis.proximity)]
        assert detected and not any(matches(a, label, analysis.proximity) for a in single['anomalies'])
        points = [p for p in single['annotated'] if first <= p['date'] <= last]
        dates = [date.fromisoformat(p['date']) for p in points]
        values = np.array([p['value'] for p in points])
        thresholds = np.array([np.nan if p['threshold'] is None else p['threshold'] for p in points])
        ax.plot(dates, values, color='black', lw=.8, label='Time series', zorder=3)
        ax.plot(dates, thresholds, color='#0088ff', ls='--', lw=.8, label='Z-Score + labels threshold')
        by_date = {p['date']:p for p in ensemble['annotated']}
        for n, anomaly in enumerate(detected):
            selected = [p for p in points if anomaly['start'] <= p['date'] <= anomaly['end']]
            x = [date.fromisoformat(p['date']) for p in selected]
            y = [p['value'] for p in selected]
            t = [by_date[p['date']]['threshold'] for p in selected]
            assert all(v is not None for v in t)
            ax.plot(x, t, color='#008000', ls='--', lw=1,
                    label='Ensemble + labels IF threshold' if n == 0 else None)
            ax.fill_between(x, t, y, color='#ff9999', alpha=.8,
                            label='Ensemble detected impact' if n == 0 else None)
            ax.axvline(date.fromisoformat(anomaly['start']), color='#9900ff', lw=.7, ls='--',
                       label='Ensemble onset' if n == 0 else None)
        ax.axvline(date.fromisoformat(when), color='#777777', lw=.8, ls='-.', label='Event label date')
        ax.set_title(title + ' (2025)', fontsize=9, loc='left')
        ax.set_ylabel('Search volume / threshold')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_xlim(date.fromisoformat(first),date.fromisoformat(last))
        ax.set_ylim(bottom=0)
        ax.grid()
        evidence.append(dict(country=country, label_date=when,
            ensemble_starts=[a['start'] for a in detected], single_detected=False,
            first=first, last=last))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=7, bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.19,1,1), w_pad=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    analysis = Analysis(root, root/'contextualization.json', args.cache.resolve())
    evidence = plot(analysis, root/'figures/detection-cases.pdf')
    result = read(root/'analysis.json')
    result['detection_cases'] = evidence
    write(root/'analysis.json', result)
    print(evidence)


if __name__ == '__main__':
    main()
