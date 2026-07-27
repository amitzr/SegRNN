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

**Results:** qualitatively **terrible** (worst-in-batch) — exact figures
not yet recorded here, pending the full per-horizon table.

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

**Results:** qualitatively **slightly worse** — exact figures not yet
recorded here, pending the full per-horizon table.

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

**Results:** qualitatively **around the same** as the reconstruction —
exact figures not yet recorded here, pending the full per-horizon table.
Notably, this contradicts the prediction made in this document's own
Discussion (below) that bidirectional, as the heaviest strand, was "the
more likely of the two [depth, direction] to regress clearly" — instead
it's depth (strand 10) that regressed, while the structurally heavier
bidirectional strand came in roughly neutral.

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

**Results:** qualitatively **bad** — exact figures not yet recorded here,
pending the full per-horizon table. Matches the prior cited in advance
(ISMRNN's reported conv regressions).

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

**Results:** qualitatively **better than `affine=False`, but still worse
than the reconstruction** — exact figures not yet recorded here, pending
the full per-horizon table. This is a clean partial confirmation of the
strand's own hypothesis: the affine correction genuinely helps (moves the
result back toward the reconstruction, as predicted), it just isn't
enough to overcome whatever makes RevIN a poor fit for this
dataset/architecture in the first place (`docs/stage2_revin_attn_draft.md`
strand 3's own diagnosis — stale/diluted full-window anchor vs. the
default's last-value anchor).

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

**Results:**

| Horizon | Config | MSE | Δ MSE | MAE | Δ MAE |
|---|---|---|---|---|---|
| 336 | Reconstruction | 0.4233 | — | 0.4327 | — |
| 336 | huber | 0.4271 | +0.9% | 0.4306 | -0.5% |
| 336 | huber + YJ | 0.3983 | -5.9% | 0.4526 | +4.6% |
| 336 | blend | 0.4258 | +0.6% | 0.4306 | -0.5% |
| 336 | blend + YJ | 0.3983 | -5.9% | 0.4530 | +4.7% |
| 720 | Reconstruction | 0.4657 | — | 0.4719 | — |
| 720 | huber | 0.4453 | **-4.4%** | 0.4563 | **-3.3%** |
| 720 | huber + YJ | 0.4448 | -4.5% | 0.4865 | +3.1% |
| 720 | blend | 0.4597 | -1.3% | 0.4662 | -1.2% |
| 720 | blend + YJ | 0.4508 | -3.2% | 0.4884 | +3.5% |

**Reading the result — the most promising strand across both documents.**
Two distinct findings, one per horizon:

**H=720: `huber` alone (no Yeo-Johnson) improves *both* MSE (-4.4%) and
MAE (-3.3%) simultaneously.** This is the first strand in the entire
project — across all 14 strands tested in `docs/stage2_new_strands_draft.md`
and here — with a clean win on both point-forecast metrics at once, no
trade-off. `d_model=256` was a free efficiency win with a small accuracy
cost; Yeo-Johnson was an MSE win with an MAE cost; this is a plain loss-
function swap, zero new parameters, that wins outright. `blend` alone is
a smaller version of the same clean-win pattern (-1.3% / -1.2%) — same
direction, less pronounced than `huber`.

**H=336: `huber`/`blend` alone are small and balanced, not dramatic** —
MSE roughly flat (+0.9%, +0.6%), MAE a small genuine improvement (-0.5%
both). Not the same clean win as H=720, but not a trade-off either —
the closest thing to "no downside" this project has produced at this
horizon.

**Does Yeo-Johnson compound with a robust loss, or compete with it? —
compete, and clearly so at H=720.** Adding `--power_transform 1` on top
of `huber` barely changes MSE at H=720 (-4.4% -> -4.5%, essentially
identical) but flips MAE from a -3.3% win into a +3.1% loss — a ~6.4
point swing for no additional MSE benefit. Same pattern for `blend`
(+3.5% MAE vs. -1.2% alone). At H=336, adding Yeo-Johnson pushes MSE
further down (-5.9%, beyond either alone) but reproduces the same MAE
cost (+4.6-4.7%) that plain Yeo-Johnson itself has
(`docs/stage2_new_strands_draft.md` strand 5: +6.7% MAE at H=336). Put
together: Huber/blend and Yeo-Johnson appear to be **two different
answers to the same problem** (MSE's outlier-sensitivity) rather than
independent, additive fixes — once a robust loss is already handling
that job, adding Yeo-Johnson on top doesn't help further and actively
costs MAE. **The practical takeaway is `huber` loss alone, without
Yeo-Johnson, not the combination.**

---

## Discussion

**Checking the prediction made before running anything** (stated below,
before results existed): three parts held, one was wrong, one was mixed.

- FFT repeats the "inject into `h_n`" failure pattern — **confirmed**,
  and more strongly than expected ("terrible" is the qualitative low
  point of this whole batch, consistent with calendar features/attention/
  ROCKET all failing the same way regardless of feature source).
- Conv embedding regresses, per the ISMRNN prior — **confirmed** ("bad").
- RevIN affine=True is a partial fix, not a full one — **confirmed
  exactly as hypothesized**: better than `affine=False`, still worse than
  no RevIN at all.
- Encoder depth and bidirectional both add capacity, predicted to
  regress or show a small effect, with bidirectional (the heavier one)
  "more likely to regress clearly" — **half right**. Depth did regress
  (slightly). Bidirectional, the *heavier* of the two by parameter count,
  came in roughly neutral instead — the opposite of which one was
  predicted to fail more clearly. Capacity alone doesn't predict outcome
  as cleanly as the rest of this project's pattern suggested; *how* the
  capacity is used seems to matter as much as *how much* is added.
- Loss function was flagged as "a genuinely open question... worth going
  in without a negative prior" — technically correct, but this
  undersells it. It turned out to be the best result in either document:
  `huber` alone at H=720 is the first strand in the whole project to
  improve *both* MSE and MAE with no trade-off at all.

**How this batch fits the wider project.** The four strands that added
information or capacity into the architecture itself (FFT, conv
embedding, encoder depth, bidirectional) landed exactly where the
established pattern predicted — failure or, at best, neutral — extending
that signature to eight strands now across four unrelated information
sources (calendar, attention, ROCKET, FFT) and three unrelated capacity
axes (`d_model`, encoder depth, bidirectional). RevIN affine=True
confirms the more nuanced reading from `docs/stage2_revin_attn_draft.md`:
the *specific implementation* tested there was fixable in the predicted
direction, even though the underlying dataset/architecture mismatch
remains. The loss-function strand is the real headline: it's the second
strand in the whole project (after `d_model=256`) with an unambiguous
win, and the *first* with no trade-off attached at all — and it argues
against combining every positive-seeming idea together, since stacking
Yeo-Johnson onto Huber turned out to make H=720 strictly worse on MAE for
no MSE gain. The practical recommendation for the final report is
`huber` loss alone as the strongest single Stage 2 change found, ahead of
Yeo-Johnson and ahead of the `d_model` efficiency win on raw accuracy
(though `d_model=256` remains the strongest result if efficiency, not
just accuracy, is the goal).

**Limitations:** strands 9-13's headline verdicts above are qualitative,
reported by the user from the notebook's live output rather than
transcribed into this document as exact per-horizon numbers yet — worth
pulling the precise tables in before finalizing the report, same standard
every other strand in this project has been held to. Strand 14's numbers
are exact (H=336, H=720 only, single seed each, no seed-variance check
yet — unlike the calendar-feature finding, not yet confirmed real vs.
noise).
