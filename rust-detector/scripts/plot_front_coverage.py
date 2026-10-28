"""Plot held-out coverage at training-only positions on saved Pareto fronts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from paper_plot_style import apply_paper_style


def plot(rows, output, mode):
    if not rows or any(r['test_labels'] == 0 for r in rows):
        raise ValueError('Coverage curves require test labels')
    apply_paper_style()
    pairs = ([('original', 'labels', 'Single detector'),
              ('ensemble', 'labels_ensemble', 'Ensemble')]
             if mode == 'labels' else
             [('original', 'ensemble', 'Unconstrained'),
              ('labels', 'labels_ensemble', 'Label-constrained')])
    legend = ['Unconstrained', 'Label-constrained'] if mode == 'labels' else ['Single detector', 'Ensemble']
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), layout='constrained')
    for col, (first, second, title) in enumerate(pairs):
        for line, variant in enumerate([first, second]):
            points = sorted([r for r in rows if r['variant'] == variant], key=lambda r:r['front_fraction'])
            for row, (x, y, xlabel) in enumerate([
                ('test_anomalies', 'raw_coverage', 'Test anomalies'),
                ('notifications', 'retained_coverage', 'Alerts')]):
                ax=axes[row,col]
                ax.plot([p[x] for p in points], [p[y] for p in points],
                        marker='o', markersize=3, linewidth=1, label=legend[line],
                        color=['#245a81','#c47726'][line], linestyle=['-','--'][line])
                ax.set(xlabel=xlabel, ylabel='Test country-label coverage', ylim=(0,1.03))
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(alpha=.4)
                ax.set_title(title)
        axes[0,col].legend()
    countries=int(rows[0]['countries']); labels=int(rows[0]['test_labels'])
    fig.supxlabel(f'{countries} countries; {labels} test labels',fontsize=8)
    fig.savefig(output/f'{mode}-coverage.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(args.input.open()))
    for r in rows:
        for k in r:
            if k!='variant':r[k]=float(r[k]) if r[k] else None
    for mode in ['labels','ensemble']:plot(rows,args.output,mode)


if __name__=='__main__':main()
