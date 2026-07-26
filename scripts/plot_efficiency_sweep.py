"""
Generate the accuracy-vs-parameter-count efficiency sweep figure for the
Stage 2 report (docs/stage2_efficiency_draft.md).

MSE values are from the notebook's Part 7 (d_model in {512,256,128,64},
all four horizons, run_longExp.py unmodified). Parameter counts are
computed analytically from models/SegRNN.py's architecture (GRU +
Linear + PE layers) rather than measured, since the sweep's wall-clock
ms/sample timings turned out to be noisy/non-monotonic (single-pass
Colab GPU timing, no warm-up) -- parameter count is exact and
deterministic instead.

Usage: python scripts/plot_efficiency_sweep.py
Output: results/figures/efficiency_params.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np

HORIZONS = [96, 192, 336, 720]
D_MODELS = [512, 256, 128, 64]
SEG_LEN = 24
ENC_IN = 7

# measured on Colab T4, notebook Part 7 (run_longExp.py, unmodified, seed=2024)
MSE = {
    96:  {512: 0.3510, 256: 0.3557, 128: 0.3677, 64: 0.3848},
    192: {512: 0.3925, 256: 0.3956, 128: 0.4007, 64: 0.4219},
    336: {512: 0.4233, 256: 0.4248, 128: 0.4304, 64: 0.4445},
    720: {512: 0.4657, 256: 0.4881, 128: 0.4915, 64: 0.4950},
}


def param_count(d_model, seg_num_y):
    """Exact SegRNN parameter count (channel_id=1, revin=0), matching
    models/SegRNN.py's layer definitions."""
    value_embedding = d_model * (SEG_LEN + 1)
    gru = 6 * d_model ** 2 + 6 * d_model
    pos_emb = seg_num_y * (d_model // 2)
    channel_emb = ENC_IN * (d_model // 2)
    predict = SEG_LEN * (d_model + 1)
    return value_embedding + gru + pos_emb + channel_emb + predict


PARAMS = {h: {d: param_count(d, h // SEG_LEN) for d in D_MODELS} for h in HORIZONS}

COLORS = {96: '#2a78d6', 192: '#008300', 336: '#e87ba4', 720: '#eda100'}
INK_PRIMARY, INK_SECONDARY, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
GRIDLINE, BASELINE_AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for h in HORIZONS:
        xs = [PARAMS[h][d] for d in D_MODELS]
        ys = [MSE[h][d] for d in D_MODELS]
        ax.plot(xs, ys, marker='o', color=COLORS[h], label=f'H={h}', linewidth=2, markersize=6)
        for d, x, y in zip(D_MODELS, xs, ys):
            if d in (512, 64):
                ax.annotate(f'd={d}', (x, y), textcoords='offset points', xytext=(4, 4),
                            fontsize=7, color=INK_MUTED)

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (log scale)', color=INK_SECONDARY)
    ax.set_ylabel('MSE', color=INK_SECONDARY)
    ax.set_title('SegRNN on ETTh1 — accuracy vs. parameter count (d_model sweep)',
                  color=INK_PRIMARY, fontsize=12, loc='left')
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(BASELINE_AXIS)
    ax.spines['bottom'].set_color(BASELINE_AXIS)
    ax.tick_params(axis='both', which='both', length=0, colors=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc='upper right')

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'efficiency_params.png')
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
