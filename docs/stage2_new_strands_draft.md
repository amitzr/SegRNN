# Stage 2 report content — Yeo-Johnson, AIC/BIC, ensembling, ROCKET (draft)

Draft content for four more independent Stage 2 strands, alongside the
calendar-feature work (`docs/stage2_report_draft.md`), the efficiency
sweep (`docs/stage2_efficiency_draft.md`), and RevIN/Attention
(`docs/stage2_revin_attn_draft.md`). Sources: notebook Parts 0a–0d,
`data_provider/data_loader.py`, `models/SegRNNRocket.py`,
`results/runs.csv`. **Results tables below are placeholders — filled in
after Parts 0a–0d are run on Colab.**

Each strand is tested independently against the same reconstruction
baseline used throughout (`RECON_BASELINE` in the notebook), not stacked
with each other or with the calendar-feature/RevIN/Attention strands —
same reasoning as `docs/stage2_revin_attn_draft.md`'s Discussion: stacking
an untested idea onto an already-negative strand would make it impossible
to attribute the result to either one.

---

## Strand 5: Yeo-Johnson power transform (preprocessing)

**What changed:** `data_provider/data_loader.py`'s `Dataset_ETT_hour`
gained a `power_transform` flag (`run_longExp.py --power_transform 1`, off
by default). When on, it fits `sklearn.preprocessing.PowerTransformer
(method='yeo-johnson', standardize=True)` on the train split only, in
place of the plain `StandardScaler` — a drop-in swap, not an addition (the
transform's `standardize=True` already zero-means/unit-variances the
transformed values, so nothing downstream changes shape-wise). With the
flag off, behavior is byte-identical to the plain reconstruction.

**Why it should help:** `docs/Pre-precessing.pdf`'s preprocessing slide
lists Yeo-Johnson as the standard fix for heteroscedasticity and
non-normality in a feature — unlike Box-Cox, it's defined for negative
values, which matters here since ETTh1's `OT` and load channels take
negative values. `StandardScaler` only recenters/rescales; it doesn't
address skew in the *distribution* of each channel, which the paper's
Table I load statistics suggest is plausible for at least a couple of
ETTh1's seven channels.

**Results:** pending — run notebook Part 0a.

---

## Strand 6: AIC/BIC post-hoc analysis of the `d_model` sweep

**What changed:** nothing — this is a zero-cost re-analysis of Part 7's
already-completed `d_model` sweep (`docs/stage2_efficiency_draft.md`), no
new training. `docs/Time-Series Forecasting.pdf`'s model-selection slide
gives the formal criteria AIC = n·ln(MSE) + 2k and BIC = n·ln(MSE) +
k·ln(n) (Gaussian-error regression form) — the quantitative version of
"prefer the simplest model that explains the data well," the same
principle behind the same deck's Theta-method note that simple methods
often outperform complex ones. `n` = test windows × horizon × channels for
each split; `k` = SegRNN's exact analytical parameter count
(`scripts/plot_efficiency_sweep.py`'s formula).

**Why this is worth doing:** the efficiency sweep's headline (`d_model`
=256 keeps 99.65% of accuracy at 25.5% of the parameters at H=336) was
read informally, off a "knee" in an accuracy-vs-params plot. AIC/BIC put a
number on the same trade-off and can disagree with that informal read —
worth knowing either way, and free to check.

**Results:** pending — run notebook Part 0b (needs `results/runs.csv` to
already have the Part 7 sweep rows, which it does).

---

## Strand 7: Multi-seed prediction ensembling

**What changed:** `exp/exp_main.py`'s `test()` now optionally saves raw
`pred.npy`/`true.npy` (`--save_preds 1`, off by default, no behavior
change otherwise). The notebook reruns the reconstruction at 3 seeds
(2021, 2022, 2024) with `--save_preds 1`, averages the three seeds'
predictions element-wise, and scores the *averaged prediction* — not the
same thing as Part 6's seed-variance check, which averages the *metrics*
across seeds rather than the predictions.

**Why it's here:** not lecture-grounded (unlike the other three strands in
this document) — flagged honestly as a general technique rather than one
from `docs/Pre-precessing.pdf` or `docs/Time-Series Forecasting.pdf`.
Cheap to test given Part 6 already established the seed-to-seed spread at
these two horizons.

**Results:** pending — run notebook Part 0c (2 horizons: 336, 720, same
scope-narrowing as Part 6, same compute-budget reason).

---

## Strand 8: ROCKET-style random-convolutional features (`SegRNNRocket`)

**What changed:** `models/SegRNNRocket.py` adds a fixed (untrained, buffer
-registered) bank of random 1D convolution kernels over the raw
(last-value-normalized) lookback window, per channel — Max and PPV
(proportion of positive values) pooling per kernel, the two statistics the
ROCKET paper uses. A trainable `Linear` projects this feature vector to
`d_model` and adds it to `h_n` before decoding:
`h_n_aug = h_n + rocket_proj(rocket_features)`. Everything else (segment
partition, value embedding, GRU encoding, PMF decode, normalization) is
identical to `models/SegRNN.py`.

**Why it should help, and the caveat:** `docs/Pre-precessing.pdf`'s
feature-engineering slide lists ROCKET (random convolutional kernels) as a
window-based feature technique. ROCKET is a classification-era method —
there's no textbook precedent for using it inside a forecaster, so this is
an honest adaptation, not a documented one; it's the most speculative of
the four strands in this document. It's also a direct repeat of the
"inject extra information into `h_n`" pattern that all three
calendar-feature attempts used and that regressed every time
(`docs/stage2_report_draft.md`) — testing it with a completely different,
non-calendar information source (features of the raw series itself) is an
honest, independent check of whether that earlier finding is
calendar-specific or a more general property of this architecture/dataset.

**Results:** pending — run notebook Part 0d.

---

## Discussion

To be written once all four strands have real results. If the pattern from
`docs/stage2_revin_attn_draft.md` holds (every "add information/richness"
strand regressed, only "reduce capacity" helped), the expected prior is:
Yeo-Johnson and ROCKET regress or are neutral (both add/reshape
information the model has to accommodate), ensembling is a coin flip
(averaging noise across seeds should help a little regardless of the
underlying architecture question), and AIC/BIC either confirms or
sharpens the `d_model=256` finding rather than overturning it. Worth
stating as a prediction now, before the runs happen, precisely so it's
falsifiable rather than written after the fact.
