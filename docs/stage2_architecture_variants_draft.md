# Stage 2 report content — FFT, encoder depth/direction, value embedding, normalization, loss function (draft)

Draft content for six more independent Stage 2 strands, alongside
`docs/stage2_report_draft.md` (calendar features), `docs/stage2_efficiency_draft.md`
(`d_model` sweep), `docs/stage2_revin_attn_draft.md` (RevIN, Attention),
and `docs/stage2_new_strands_draft.md` (Yeo-Johnson, AIC/BIC, ensembling,
ROCKET). Sources: `notebooks/colab_runner_stage2b.ipynb`,
`models/SegRNNFFT.py`, `models/SegRNNDeepEncoder.py`, `models/SegRNNBidir.py`,
`models/SegRNNConvEmbed.py`, `models/SegRNNRevINAffine.py`, `exp/exp_main.py`
(loss criterion), `utils/tools.py` (`BlendLoss`).

Each strand is tested independently against the same reconstruction
baseline used throughout (`RECON_BASELINE`), not stacked with each other
or with any prior strand — same reasoning as every earlier document's
Discussion section. Run in a second, self-contained notebook
(`colab_runner_stage2b.ipynb`) rather than prepended to the first —
kept separate given how much is already in `colab_runner.ipynb`.

---

## Strand 9: top-K FFT-magnitude features (`SegRNNFFT`)

**What changed:** a deterministic (no learnable/random parameters of its
own) feature extractor takes the top-K largest-magnitude FFT bins of the
raw (last-value-normalized) look-back window, per channel; a trainable
`Linear` projects the K magnitudes to `d_model` and adds them to `h_n`
before decoding — the same injection point `SegRNNRocket` used, a
different feature source. Run twice per horizon: `--power_transform 0`
and `1`.

**Why both with and without Yeo-Johnson — the logic, not just combinatorics:**
Yeo-Johnson is a nonlinear pointwise transform of the amplitude; it does
not commute with the Fourier transform (nonlinear amplitude transforms
change harmonic content, the same reason nonlinear amplifiers create
distortion/new frequencies in signal processing). So "FFT on raw values"
and "FFT on Yeo-Johnson-transformed values" are genuinely different
features, and this is a legitimate follow-up rather than stacking onto an
already-failed strand — Yeo-Johnson is the one strand in the whole
project with a measured positive effect (`docs/stage2_new_strands_draft.md`
strand 5).

**Results:** pending — run notebook Part 1.

---

## Strand 10: encoder depth — stacked GRU (`SegRNNDeepEncoder`)

**What changed:** `--encoder_layers 2` stacks the encoder GRU instead of
the original single layer. The PMF decode step's hidden-state reshape
(previously hardcoded to a leading dim of 1) is generalized to
`encoder_layers`; the shared-cell trick (same GRU used for encode and
decode) still applies, since `nn.GRU`'s forward pass works the same way
regardless of `num_layers`.

**Why this is a different question from the paper's own ablation:** the
paper's segment-length sweep (Fig. 6) varies `n = L/w`, the number of
sequential *recurrent steps* a single-layer GRU unrolls over, and already
found the sweet spot there (`w=48`, or `24` for ETTh1). This strand
instead stacks additional GRU *layers* processing the same `n`-step
sequence — untested by the paper and untested elsewhere in this project.
Depth and width (`d_model`) are different levers, so the `d_model`
sweep's "less is more" finding doesn't automatically transfer — but it's
the same *direction* of change (more parameters), so the honest prior
going in is skepticism, not a blank slate.

**Results:** pending — run notebook Part 2.

---

## Strand 11: encoder direction — bidirectional (`SegRNNBidir`)

**What changed:** the encoder becomes a bidirectional GRU (`CLAUDE.md`'s
own unimplemented improvement B); forward+backward final states are
concatenated (`2d`) and projected back to `d` before decoding. Since a
bidirectional encoder can't be reused for the PMF decode step (decoding
is a single forward step, there's no "reverse direction" to run over),
this strand necessarily breaks SegRNN's shared-cell trick and adds a
full second, decode-only unidirectional GRU on top.

**Why it should help, and the honest cost:** the whole look-back window
is past data by encoding time, so bidirectional processing isn't
leakage, and it could capture patterns a forward-only pass misses. But
this is the heaviest strand tested in the project by parameter count —
roughly doubles encoder parameters (bidirectional GRU) *and* adds a full
second GRU's worth of decode-only parameters, on top of a small
projection layer. Given every added-capacity strand so far has regressed
and AIC/BIC formally rejects even `d_model=512`'s parameter count
(`docs/stage2_new_strands_draft.md` strand 6), this needs a strong effect
to be worth its cost.

**Results:** pending — run notebook Part 3.

---

## Strand 12: value embedding — within-segment `Conv1d` (`SegRNNConvEmbed`)

**What changed:** replaces `Linear(seg_len -> d_model) + ReLU` with a
`Conv1d` (kernel width `--conv_kernel_size`, default 5 — narrower than
`seg_len`) applied within each segment, followed by mean-pooling over
time, producing the same one-vector-per-segment shape the original
Linear did.

**Why this is a genuinely different computation, not a relabeling:** a
`Conv1d` with `kernel_size == seg_len` (one application per segment, no
sliding) would be mathematically equivalent to the original `Linear` —
no new inductive bias. Using a narrower kernel makes the conv slide
across multiple positions within a segment, giving the embedding step a
translation-equivariant, local-pattern bias that a single `Linear` layer
(which treats the `w` values as a flat vector with an independent weight
per position) does not have.

**Prior evidence, cited honestly in advance:** `CLAUDE.md` already flags
this as the higher-risk of its two listed architecture candidates, citing
ISMRNN (arXiv 2407.10768) reporting conv hurting on all datasets except
Weather — going in with that prior, not a blank slate.

**Results:** pending — run notebook Part 4.

---

## Strand 13: normalization — RevIN with `affine=True` (`SegRNNRevINAffine`)

**What changed:** identical to `models/SegRNN.py` except the RevIN
instantiation's `affine` argument (`False` -> `True`) — adds RevIN's
learnable per-channel post-normalization scale and shift, which the
already-tested configuration disabled entirely.

**Why:** RevIN with `affine=False` regressed at every horizon
(`docs/stage2_revin_attn_draft.md` strand 3, +3.1% to +4.6% MSE), and
that write-up's own limitations section names `affine=True` as the
specific untested variant that "might behave differently, even on
ETTh1" — this strand is that direct follow-up, not a new idea.

**Results:** pending — run notebook Part 5 (compared against both the
reconstruction and the already-measured `affine=False` numbers).

---

## Strand 14: loss function — Huber and blended MSE+MAE, with and without Yeo-Johnson

**What changed:** `exp/exp_main.py`'s `_select_criterion` gains two new
options (`--loss huber`, delta configurable via `--huber_delta`; `--loss
blend`, `alpha*MSE + (1-alpha)*MAE` via `utils/tools.py`'s new
`BlendLoss`, alpha configurable via `--blend_alpha`). No new model file —
plain `SegRNN`, just a different training objective. Each loss tried with
`--power_transform 0` and `1` — 4 combinations, scoped to 2 horizons
(336, 720) to keep the run count reasonable (same reasoning as the first
notebook's Part 6 seed-variance check and Part 0c ensembling).

**Why this is worth doing:** targets the exact mechanism behind Yeo-
Johnson's own result (`docs/stage2_new_strands_draft.md` strand 5): MSE's
outlier-sensitivity is *why* compressing outlier scale (Yeo-Johnson)
improved MSE while hurting MAE. A loss-level fix goes at the same root
cause a different way — Huber and the MSE/MAE blend are both explicitly
designed to be less outlier-dominated than plain MSE without discarding
outlier information the way a preprocessing transform does. Testing both
with and without Yeo-Johnson checks whether a loss-level fix makes the
preprocessing-level fix redundant, additive, or actively conflicting
(e.g., Huber's own outlier-robustness acting on already-compressed
values may behave differently than on raw ones).

**Results:** pending — run notebook Part 6.

---

## Discussion

To be written once all six strands have real results. Given the
established pattern (`docs/stage2_new_strands_draft.md`'s Discussion:
every strand that added information or architectural richness regressed;
only reducing capacity or reshaping the input distribution without adding
information either succeeded or produced an honest trade-off), a stated
prior before running anything, so it stays falsifiable:
- Encoder depth and bidirectional encoding both add capacity in the
  direction AIC/BIC already rejected — expect regression or, at best,
  a small/noise-scale effect, with bidirectional (the heaviest strand)
  the more likely of the two to regress clearly.
- Conv embedding: `CLAUDE.md`'s own cited prior work (ISMRNN) predicts
  regression except possibly on Weather-like data, not ETTh1.
- FFT features repeat the "inject into `h_n`" pattern that has failed
  for three prior information sources (calendar, attention, ROCKET) —
  expect the same outcome, though FFT is a more classically appropriate
  feature source than ROCKET was, so less confident in this one than in
  the capacity-adding strands above.
- RevIN affine=True is the one strand here testing a *fix* to an
  already-diagnosed problem (the disabled affine correction), not a new
  capacity addition — the closest thing to a plausible win in this batch.
- Loss function (Huber/blend) is, like Yeo-Johnson, not a capacity change
  at all — a genuinely open question, and the one other strand besides
  RevIN-affine worth going in without a negative prior.
