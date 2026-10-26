"""Shared typography and axes for CenAlert manuscript figures."""
import matplotlib.pyplot as plt


def apply_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',
        'axes.titlesize': 9,
        'axes.titleweight': 'normal',
        'axes.labelsize': 8,
        'axes.linewidth': .5,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.axisbelow': True,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'xtick.major.width': .5,
        'ytick.major.width': .5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'grid.color': '#b0b0b0',
        'grid.alpha': .6,
        'grid.linestyle': '--',
        'grid.linewidth': .4,
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.framealpha': 1,
        'legend.edgecolor': '#cccccc',
        'legend.borderpad': .25,
        'lines.linewidth': .8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
