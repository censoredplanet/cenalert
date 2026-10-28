"""Plot saved held-out detection cases in the style of CenAlert Figure 4."""
import argparse
import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze_paper_results import Analysis, event_text, label_kind
from batch_ensemble_tuning import label_applies
from review_contextualization import matches, read, write
from paper_plot_style import apply_paper_style

CASES = (
    ('RU', '2025-06-09', '2025-05-11', '2025-06-23', 'Russia: Cloudflare throttling'),
    ('TM', '2025-07-21', '2025-06-22', '2025-07-28', 'Turkmenistan: VPN blocking'),
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


def threshold_replay(analysis, country, first, last, binary):
    """Replay saved parameters with optional idle-day threshold diagnostics."""
    folder = analysis.folder(country, 'labels_ensemble')
    source = analysis.root / f'inputs/{country}-evaluation.json'
    signature = dict(parameters=read(folder/'knee-parameters.json'),
                     first=first, last=last, reference_end=analysis.cutoff,
                     binary=str(binary), binary_mtime=binary.stat().st_mtime_ns,
                     input_mtime=source.stat().st_mtime_ns)
    cached = analysis.cache/f'{country}-case-thresholds.json'
    previous = read(cached) if cached.exists() else {}
    if previous.get('signature') == signature:
        result = previous['result']
    else:
        command = [str(binary), 'detect', '--algorithm', 'ensemble',
                   '--input', str(source), '--parameters', str(folder/'knee-parameters.json'),
                   '--reference-end', analysis.cutoff, '--include-annotated',
                   '--thresholds-from', first, '--thresholds-through', last]
        result = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        write(cached, dict(signature=signature, result=result))
    baseline = replay(analysis, country, 'labels_ensemble')
    assert result['anomalies'] == baseline['anomalies'], country
    assert result['scale_factor'] == baseline['scale_factor'], country
    assert [{k:v for k,v in p.items() if k != 'diagnostic_threshold'}
            for p in result['annotated']] == baseline['annotated'], country
    return result


def plot_threshold(point):
    return point['threshold'] if point['threshold'] is not None else point.get('diagnostic_threshold')


def restriction_cases(analysis):
    records = {(r['country'], r['variant']): r for r in analysis.context['countries']
               if r['selection'] == 'knee'}
    cases = []
    for label in analysis.labels:
        if label['scope'] != 'country' or not analysis.first_test <= label['date'] <= analysis.config['test_end']:
            continue
        for (country, variant), row in records.items():
            if variant != 'labels_ensemble' or not label_applies(label, country):
                continue
            detected = [a for a in row['raw_anomalies'] if matches(a, label, analysis.proximity)]
            single = records[country, 'labels']['raw_anomalies']
            if not detected or any(matches(a, label, analysis.proximity) for a in single):
                continue
            if label_kind(label) != 'restriction_related':
                continue
            first = min(a['start'] for a in detected)
            last = max(a['end'] for a in detected)
            cases.append((country, label['date'], str(date.fromisoformat(first)-timedelta(days=30)),
                          min(analysis.config['test_end'], str(date.fromisoformat(last)+timedelta(days=7))),
                          event_text(label)))
    return sorted(set(cases))


def plot(analysis, destination, binary, cases=CASES, pdf=None):
    apply_paper_style()
    fig, axes = plt.subplots(2, len(cases), figsize=(7, 4.15), sharex='col', sharey='col', squeeze=False)
    evidence = []
    for col, (country, when, first, last, description) in enumerate(cases):
        single = replay(analysis, country, 'labels')
        ensemble = threshold_replay(analysis, country, first, last, binary)
        label = dict(scope='country', date=when)
        detected = [a for a in ensemble['anomalies'] if matches(a, label, analysis.proximity)]
        assert detected and not any(matches(a, label, analysis.proximity) for a in single['anomalies'])
        for row, output in enumerate((single, ensemble)):
            ax = axes[row, col]
            points = [p for p in output['annotated'] if first <= p['date'] <= last]
            dates = [date.fromisoformat(p['date']) for p in points]
            values = np.array([p['value'] for p in points])
            thresholds = np.array([np.nan if plot_threshold(p) is None else plot_threshold(p) for p in points])
            # Croston's sparse path has no dense inverse-score threshold on idle dates.
            ax.plot(dates, values, color='black', lw=1.2, zorder=3)
            ax.plot(dates, thresholds, color='blue', ls='--', lw=.9)
            for anomaly in output['anomalies']:
                if anomaly['end'] < first or anomaly['start'] > last:
                    continue
                selected = [p for p in points if anomaly['start'] <= p['date'] <= anomaly['end']]
                x = [date.fromisoformat(p['date']) for p in selected]
                y = [p['value'] for p in selected]
                t = [p['threshold'] for p in selected]
                assert all(v is not None for v in t)
                if len(x) == 1:
                    ax.plot(x, y, 'o', color='red', markersize=3, zorder=4)
                else:
                    ax.fill_between(x, t, y, color='red', alpha=.4)
                ax.axvline(date.fromisoformat(anomaly['start']), color='darkorchid', lw=.8, ls='--')
            ax.text(.025, .93, f'({chr(97 + row*len(cases) + col)})', transform=ax.transAxes,
                    va='top', fontsize=9)
            ax.set_ylabel('Search Volume')
            ax.set_xticks([dates[i] for i in np.linspace(0, len(dates)-1, 5, dtype=int)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.set_xlim(dates[0], dates[-1])
            ax.grid(which='both', color='black', linestyle='--', alpha=.3)
            if row == 1:
                ax.set_xlabel('Date')
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].yaxis.set_major_locator(MaxNLocator(nbins=5))
        evidence.append(dict(country=country, label_date=when,
            ensemble_starts=[a['start'] for a in detected], single_matching_onset=False,
            single_overlapping_intervals=[dict(start=a['start'], end=a['end'])
                for a in single['anomalies'] if a['start'] <= when <= a['end']],
            event=description, first=first, last=last))
    handles = [Line2D([], [], color='black', lw=1.2, label='Search volume'),
               Line2D([], [], color='blue', ls='--', lw=.9, label='Threshold'),
               Line2D([], [], color='darkorchid', ls='--', lw=.8, label='Detected onset'),
               Patch(facecolor='red', alpha=.4, label='Detected impact')]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8, bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0,.07,1,1), w_pad=2, h_pad=.8)
    if pdf is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, bbox_inches='tight', pad_inches=.03)
    else:
        # Review pages identify cases below the plots; manuscript panels have no titles.
        import textwrap
        captions = ['Top: label-constrained Z-Score. Bottom: label-constrained ensemble.']
        captions += [f"{c} ({d}): {e}" for c,d,_,_,e in cases]
        for item in evidence:
            for interval in item['single_overlapping_intervals']:
                captions.append(f"{item['country']}: Z-Score includes the event in an interval starting "
                                f"{interval['start']}, outside the label-matching window.")
        fig.text(.02, -.02, '\n'.join(textwrap.fill(t, 115) for t in captions), va='top', fontsize=8)
        pdf.savefig(fig, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--rust-binary', type=Path,
                        default=Path(__file__).resolve().parents[1]/'target/release/cenalert-detector',
                        help='Detector binary supporting optional threshold diagnostics.')
    args = parser.parse_args()
    root = args.results.resolve()
    analysis = Analysis(root, root/'contextualization.json', args.cache.resolve())
    evidence = plot(analysis, root/'figures/detection-cases.pdf', args.rust_binary.resolve())
    result = read(root/'analysis.json')
    result['detection_cases'] = evidence
    cases = restriction_cases(analysis)
    reviewed = []
    with PdfPages(root/'figures/detection-cases-all.pdf') as pdf:
        for index in range(0, len(cases), 2):
            reviewed.extend(plot(analysis, None, args.rust_binary.resolve(), cases[index:index+2], pdf))
    result['restriction_detection_cases'] = reviewed
    write(root/'analysis.json', result)
    print(evidence)


if __name__ == '__main__':
    main()
