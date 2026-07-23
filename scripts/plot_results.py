"""
Generate comparison figures (paper's own MSE/MAE metrics): paper-reported
vs. our reconstruction vs. classical baselines vs. the Stage 2 improved
model (SegRNNTime), on ETTh1 across all four forecast horizons.

Paper and reconstruction numbers are hardcoded below (published/verified
values -- see docs/stage1_report_draft.md). Baseline (naive,
seasonal_naive) and improved-model (SegRNNTime) numbers are read from
results/runs.csv if present -- the script degrades gracefully and just
omits series that aren't in the CSV yet.

Usage: python scripts/plot_results.py
Output: results/figures/mse_comparison.png, results/figures/mae_comparison.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HORIZONS = [96, 192, 336, 720]

# Paper's Table II, ETTh1, multivariate, L=720 (docs/SegRNN_paper.pdf)
PAPER = {
    'mse': {96: 0.351, 192: 0.392, 336: 0.423, 720: 0.466},
    'mae': {96: 0.392, 192: 0.414, 336: 0.433, 720: 0.472},
}

# Our reconstruction: scripts/SegRNN/etth1.sh run on Colab (T4 GPU), seed 2024
RECON = {
    'mse': {96: 0.3510, 192: 0.3925, 336: 0.4233, 720: 0.4657},
    'mae': {96: 0.3925, 192: 0.4142, 336: 0.4327, 720: 0.4720},
}

RUNS_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'runs.csv')

# dataviz skill's validated categorical palette, first 5 slots, fixed order
COLORS = {
    'Paper': '#2a78d6',
    'Reconstruction': '#008300',
    'Naive': '#e87ba4',
    'Seasonal-naive': '#eda100',
    'Improved': '#1baf7a',
}
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE_AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'


def load_runs():
    """Read naive/seasonal_naive/SegRNNTime rows from results/runs.csv, if present.

    Returns {model_name: {horizon: {'mse':.., 'mae':..}}} for each of
    naive, seasonal_naive, SegRNNTime -- missing entries are simply absent
    (chart omits those bars/series).
    """
    out = {'naive': {}, 'seasonal_naive': {}, 'SegRNNTime': {}}
    if not os.path.exists(RUNS_CSV):
        return out
    with open(RUNS_CSV, newline='') as f:
        for row in csv.DictReader(f):
            model = row.get('model')
            if model not in out:
                continue
            if row.get('dataset') != 'ETTh1':
                continue
            try:
                horizon = int(row['horizon'])
                mse = float(row['mse'])
                mae = float(row['mae'])
            except (KeyError, ValueError):
                continue
            out[model][horizon] = {'mse': mse, 'mae': mae}
    return out


def plot_metric(metric_name, runs, out_path):
    series = [('Paper', PAPER[metric_name]), ('Reconstruction', RECON[metric_name])]
    for label, key in [('Naive', 'naive'), ('Seasonal-naive', 'seasonal_naive'), ('Improved', 'SegRNNTime')]:
        values = runs[key]
        if all(h in values for h in HORIZONS):
            series.append((label, {h: values[h][metric_name] for h in HORIZONS}))

    n_series = len(series)
    n_groups = len(HORIZONS)
    group_width = 0.8
    bar_width = group_width / n_series
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (label, values) in enumerate(series):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        heights = [values[h] for h in HORIZONS]
        bars = ax.bar(offsets, heights, width=bar_width * 0.9,
                       color=COLORS[label], label=label,
                       edgecolor=SURFACE, linewidth=0.5)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7.5,
                    color=INK_PRIMARY)

    all_labels = ['Paper', 'Reconstruction', 'Naive', 'Seasonal-naive', 'Improved']
    missing = [l for l in all_labels if l not in {s[0] for s in series}]
    subtitle = f' (pending: {", ".join(missing)})' if missing else ''

    ax.set_xticks(x)
    ax.set_xticklabels([f'H={h}' for h in HORIZONS], color=INK_SECONDARY)
    ax.set_ylabel(metric_name.upper(), color=INK_SECONDARY)
    ax.set_title(f'SegRNN on ETTh1 — {metric_name.upper()} vs. paper{subtitle}',
                 color=INK_PRIMARY, fontsize=12, loc='left')

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE_AXIS)
    ax.tick_params(axis='both', which='both', length=0, colors=INK_MUTED)

    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.12),
              ncol=n_series, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f'wrote {out_path} ({n_series} series: {[s[0] for s in series]})')


def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures'), exist_ok=True)
    runs = load_runs()
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    plot_metric('mse', runs, os.path.join(fig_dir, 'mse_comparison.png'))
    plot_metric('mae', runs, os.path.join(fig_dir, 'mae_comparison.png'))


if __name__ == '__main__':
    main()
