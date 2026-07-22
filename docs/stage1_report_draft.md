# Stage 1 report content — draft

Draft content for report sections 1–3 (per the assignment's required PDF
structure). Written to be copied into the final report and lightly edited,
not a finished document. Sources: `docs/architecture_notes.md`,
`docs/data_pipeline_audit.md`, the SegRNN paper (`docs/SegRNN_paper.pdf`),
and the Colab run transcript.

**Task setup (assignment requires stating this explicitly):** multivariate
(`--features M`), supervised, point forecasting. Dataset: ETTh1 (7
channels, hourly frequency). Look-back window `L=720`, forecast horizons
`H ∈ {96, 192, 336, 720}`, segment length `w=24`, hidden size `d=512`,
random seed `2024` (fixed, `run_longExp.py`'s `--random_seed` default).

---

## 1. Original architecture

SegRNN (Lin et al., *SegRNN: Segment Recurrent Neural Network for
Long-Term Time-Series Forecasting*, IEEE IoT Journal, 2026) reduces the
number of recurrent iterations an RNN needs for long-horizon forecasting,
via two mechanisms:

**Encoding — segment-wise iteration.** Instead of feeding a look-back
window of length `L` into an RNN one timestep at a time (`L` iterations),
the input is split into `n = L/w` non-overlapping segments of length `w`.
Each segment is projected to hidden size `d` via `Linear(w→d)` + ReLU,
then the sequence of `n` segment embeddings is passed through a
single-layer GRU — only `n` iterations instead of `L`. The final hidden
state `h_n` summarizes the whole look-back window.

**Decoding — parallel multi-step forecasting (PMF).** The traditional
approach (RMF) predicts one segment at a time and feeds each prediction
back into the RNN autoregressively (`H/w` sequential iterations, and
errors compound across steps). PMF instead builds `m = H/w` positional
embeddings — each the concatenation of a learnable relative-position
encoding `rp` (marks *which* future segment this is) and a learnable
channel-position encoding `cp` (marks *which* variate this is, used only
in multivariate mode) — and passes all `m` of them through the **same**
GRU cell **in a single parallel call**, paired with `h_n` repeated `m`
times. This yields `m` output vectors in one iteration instead of `m`
sequential ones. Each is passed through Dropout → `Linear(d→w)` and
reshaped back to the `H`-length forecast.

**Normalization:** subtract the input's last observed value before
encoding, add it back after decoding (Eq. 5–6 in the paper) — a
lightweight alternative to full instance normalization; the codebase also
supports optional RevIN for datasets with severe distribution shift.

**Loss / training objective:** plain MSE (L2) between prediction and
ground truth (Eq. 7), on the scaler-standardized values (not inverse-
transformed back to original units — see `docs/data_pipeline_audit.md`
§5). Adam optimizer, 30 epochs, learning-rate decay ×0.8 after the first
3 epochs, early stopping with patience 5.

**Key hyperparameters (as used for ETTh1, `scripts/SegRNN/etth1.sh`):**
look-back `L=720`, segment length `w=24`, GRU hidden size `d=512`, single
GRU layer, dropout `0.1`, batch size `64`, learning rate `3e-4`, channel
identifier (CP encoding) enabled.

Full block-by-block code↔paper mapping (with exact file/line references)
is in `docs/architecture_notes.md`; it also documents four ablation knobs
the released code exposes beyond what's in the paper excerpt (RNN/LSTM
cell-type option, the RMF decode alternative, optional RevIN, and
`channel_id=0` to disable CP encoding) — none of which are active in the
reference configuration used here.

---

## 2. Paper results

Paper's Table II, ETTh1, multivariate forecasting, `L=720`:

| Horizon (H) | MSE | MAE |
|---|---|---|
| 96 | 0.351 | 0.392 |
| 192 | 0.392 | 0.414 |
| 336 | 0.423 | 0.433 |
| 720 | 0.466 | 0.472 |

SegRNN ranks top-2 on 54/64 metrics across all 8 benchmark datasets in the
paper's multivariate evaluation (34 first-place results), and outperforms
GRU and DeepAR baselines by 75%/78% MSE respectively. Metric: the paper
reports MSE and MAE, computed in the model's standardized (scaled) output
space — same convention this repo's `exp/exp_main.py` uses (confirmed in
`docs/data_pipeline_audit.md`).

---

## 3. Reconstruction results

Ran the paper's own reference script (`scripts/SegRNN/etth1.sh`, unmodified
hyperparameters) end to end on Google Colab (T4 GPU), fixed seed 2024:

| Horizon (H) | Paper MSE | **Our MSE** | Paper MAE | **Our MAE** |
|---|---|---|---|---|
| 96 | 0.351 | **0.3510** | 0.392 | **0.3925** |
| 192 | 0.392 | **0.3925** | 0.414 | **0.4142** |
| 336 | 0.423 | **0.4233** | 0.433 | **0.4327** |
| 720 | 0.466 | **0.4657** | 0.472 | **0.4720** |

**Match quality:** essentially exact — differences appear only in the 4th
decimal place, consistent with ordinary GPU floating-point/cuDNN run-to-run
nondeterminism rather than any methodological gap. There is no meaningful
discrepancy to explain here: we used the paper authors' own released code,
default hyperparameters, and fixed random seed, on the same dataset split
the paper describes.

**Baseline comparison (required by the assignment):** `scripts/baselines.py`
implements naive (repeat last value) and seasonal-naive (repeat last 24h
cycle) forecasts on the identical data pipeline — same `Dataset_ETT_hour`
class, same chronological 12/4/8-month split, same `StandardScaler` fit on
train only — so these numbers are directly comparable to SegRNN's. It also
reports MASE (mean absolute scaled error), scaled by the seasonal-naive
in-sample MAE computed on the training split only, as a second metric
beyond the paper's own MSE/MAE, per the assignment's requirement to
evaluate with "at least one metric studied in class."

*[PLACEHOLDER — fill in after running the Colab notebook's baselines cell,
which appends these automatically to `results/runs.csv`]*

| Horizon | naive MSE | naive MAE | naive MASE | seasonal-naive MSE | seasonal-naive MAE | seasonal-naive MASE | SegRNN MSE | SegRNN MAE |
|---|---|---|---|---|---|---|---|---|
| 96 | | | | | | | 0.3510 | 0.3925 |
| 192 | | | | | | | 0.3925 | 0.4142 |
| 336 | | | | | | | 0.4233 | 0.4327 |
| 720 | | | | | | | 0.4657 | 0.4720 |

**Temporal evaluation protocol validity:** the split is strictly
chronological by row index (train = first 12 months, val = next 4, test =
next 8, no shuffling of the underlying series), and the `StandardScaler`
is fit only on the training slice — confirmed no leakage (full analysis in
`docs/data_pipeline_audit.md` §6). In-DataLoader batch shuffling during
training only reorders which pre-built sliding window is drawn per batch,
not the rows within a window, so it doesn't violate temporal causality.
