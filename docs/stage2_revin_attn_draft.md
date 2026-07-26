# Stage 2 report content — RevIN & Attention (draft)

Draft content for two more independent Stage 2 improvement strands,
alongside the calendar-feature work (`docs/stage2_report_draft.md`) and
the efficiency sweep (`docs/stage2_efficiency_draft.md`). Sources:
notebook Parts 8–9, `models/SegRNNAttn.py`, `results/runs.csv`.

---

## Strand 3: RevIN normalization

**What changed:** nothing new — `layers/RevIN.py` and its conditional
wiring into `models/SegRNN.py` (`if self.revin: ...`) already exist in
the repo, unused (`--revin` defaults to `0`). Switching it on replaces
SegRNN's default normalization (subtract the input window's *last
value*, add it back after decoding) with per-window z-score
standardization: `z=(x-μ)/σ` computed over the *entire* look-back window,
inverted as `x̂=zσ+μ` after decoding (`affine=False`, `subtract_last=False`
in this repo's instantiation — plain z-score, no learnable affine
transform on top).

**Why it should help:** `docs/DL_for_TS.pdf`'s "Embedding Layer" slide
teaches this exact formula as the standard fix for distribution shift in
a windowed regression setup. SegRNN's default normalization only anchors
each window to its *single last point* — noisy if that point happens to
be an outlier — while RevIN anchors to the window's full mean/variance,
a more stable statistic. ETTh1 spans ~2 years (`docs/SegRNN_paper.pdf`
Table I), so some seasonal-scale level/variance shift between train and
test is plausible.

**Results:**

| Horizon | Reconstruction MSE | RevIN MSE | Δ | Reconstruction MAE | RevIN MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | 0.3673 | +4.6% | 0.3925 | 0.4003 | +2.0% |
| 192 | 0.3925 | 0.4068 | +3.6% | 0.4142 | 0.4224 | +2.0% |
| 336 | 0.4232 | 0.4376 | +3.4% | 0.4327 | 0.4397 | +1.6% |
| 720 | 0.4656 | 0.4801 | +3.1% | 0.4719 | 0.4776 | +1.2% |

**Reading the result:** consistently worse at every horizon, by a larger
and more uniform margin than either the calendar-feature or attention
strands — this is not a marginal, seed-noise-scale effect. Two likely
reasons, both about *this repo's specific* RevIN instantiation rather
than the technique in general:
1. **`affine=False`** — this call (`RevIN(self.enc_in, affine=False,
   subtract_last=False)`, `models/SegRNN.py`) disables RevIN's learnable
   post-normalization scale/shift. Published uses of RevIN typically
   enable it, giving the model a way to correct for whatever the raw
   z-score transform gets wrong; without it, the transform is applied
   rigidly.
2. **Full-window mean vs. last value.** SegRNN's default normalization
   anchors each forecast to the window's *most recent* point — the
   causally closest, most relevant reference for near-term extrapolation.
   RevIN (`subtract_last=False` here) instead anchors to the mean of the
   *entire* 720-hour window, blending in values up to 30 days old. For a
   task where the decoder's output is added back onto this anchor point
   (`y = prediction + anchor`), a stale, diluted anchor plausibly hurts
   more than it helps, even though it's a more "statistically proper"
   normalization in the abstract.

---

## Strand 4: Attention over encoder states (`SegRNNAttn`)

**What changed:** `models/SegRNNAttn.py` keeps the encoder's full
per-segment output sequence (all `n=30` GRU hidden states, not just the
final `h_n`) instead of discarding everything but the last one. In the
PMF decode step, each of the `m` target positions' positional embedding
`pe_j` is used as a query in standard scaled dot-product attention over
the encoder's `n` states (learned key/value projections); `pe_j +
context_j` — not `pe_j` alone — is what's fed into the shared decode GRU
call. The encode/decode weight-sharing trick central to SegRNN's design
is preserved (still the same GRU cell); only the decode step's *input*
changes.

**Why it should help:** `docs/DL_for_TS.pdf`'s Attention slide diagnoses
the exact problem a single encoder context vector creates — an
"information bottleneck" — and prescribes attention as the fix. This is
the direct, structural counterpart to `docs/stage2_report_draft.md`'s
diagnosis of why all three encoder-side calendar-feature attempts
regressed: they added information that had to compete for space in
`h_n`. This strand tests the complementary hypothesis — instead of
adding information, remove the bottleneck that made adding information
costly in the first place.

**Results:**

| Horizon | Reconstruction MSE | Attention MSE | Δ | Reconstruction MAE | Attention MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | 0.3564 | +1.5% | 0.3925 | 0.3977 | +1.3% |
| 192 | 0.3925 | 0.3944 | +0.5% | 0.4142 | 0.4228 | +2.1% |
| 336 | 0.4232 | 0.4447 | +5.1% | 0.4327 | 0.4496 | +3.9% |
| 720 | 0.4656 | 0.4820 | +3.5% | 0.4719 | 0.4861 | +3.0% |

**Reading the result:** worse at every horizon, and the largest single
regression of any Stage 2 strand at H=336 (+5.1%). This is a real,
substantive finding, not a footnote: it tests the direct structural
counterpart to the calendar-feature diagnosis (bottleneck vs.
representation) and the bottleneck hypothesis does *not* pan out as a
source of recoverable headroom on ETTh1 — removing it doesn't help any
more than trying to feed more information through it did. See Discussion
for what this converges on across all four strands.

---

## Discussion

**The full Stage 2 picture — four structural hypotheses, one capacity
hypothesis:**

| Strand | Change | Best-case MSE delta vs. reconstruction |
|---|---|---|
| Calendar features (×3 attempts) | add information to encoder or decoder | +0.2% to +9.6% (always worse) |
| RevIN | swap normalization strategy | +3.1% to +4.6% (always worse) |
| Attention | remove the encoder bottleneck | +0.5% to +5.1% (always worse) |
| `d_model` sweep | **reduce** capacity | **−0.35% to +1.3%** (256 ≈ free at H≤336) |

Every attempt to give the model *more* — more information (calendar
features), a *different* way of normalizing (RevIN), or a *structurally
richer* decode mechanism (attention) — made results worse, consistently,
across four independently-motivated hypotheses. The only change that
helped was giving the model *less* (fewer parameters at `d_model=256`).
That's not a coincidence worth writing off; it's a signal, and it
directly answers a question worth asking explicitly:

**Are we just failing to out-engineer an already-minimal design?**
Very plausibly yes — and there's direct textual evidence for this in the
paper itself, not just our own results. The SegRNN paper's *own* ablation
study (Section V-D1, Table IV) finds that *overlapping* segmentation —
objectively richer, more information-preserving than the default
non-overlapping scheme — **consistently underperforms** it, for two
stated reasons: "redundant historical information" and "increased
recurrent depth" that "weakens temporal information propagation." That
is the *exact same pattern* we independently rediscovered four times over
with a completely different set of experiments: added complexity/richness
consistently loses to the leaner default. Our results don't just fail to
beat the paper — they **independently corroborate**, via different
methods, the design philosophy the paper's own authors identify as
central to why SegRNN works: minimizing recurrent depth and avoiding
redundant information, even when that "redundant" information looks
useful on paper (extra segmentation overlap, or here, extra calendar/
attention signal).

**Does this mean SegRNN can't be improved, full stop?** No — it means
this specific strategy (add richer inputs/mechanisms to *this*
architecture on *this* dataset) is the wrong lever, not that improvement
is impossible in general. Two important caveats on scope:
1. All four strands were tested only on **ETTh1**. The paper evaluates on
   eight datasets with very different characteristics — notably, the
   paper states it applies RevIN specifically "for datasets with severe
   distribution shift (e.g., Traffic)," and separately reports Traffic
   has ~17x more extreme values per channel than Electricity (23.8 vs.
   1.4) and far more than Solar-Energy (0). That is, **the paper's own
   authors already predict RevIN wouldn't be universally beneficial** —
   consistent with what we found on ETTh1 (RevIN hurt), and suggesting a
   dataset like Traffic, with genuine severe distribution shift, is where
   RevIN (and plausibly calendar features tied to its rush-hour/weekday
   irregularity) would be expected to help instead. This isn't tested
   here, but it's a directly falsifiable, well-motivated extension rather
   than a vague "try more datasets" suggestion.
2. The techniques we tried are specific implementations of general ideas
   (e.g., one particular attention formulation, one particular RevIN
   configuration with `affine=False`). A different formulation of the
   same idea (e.g., RevIN with `affine=True`, restoring the learnable
   correction this repo's instantiation disables) might behave
   differently, even on ETTh1.

**Limitations:** all four strands tested on ETTh1 only, single seed for
RevIN/Attention (unlike the calendar-feature work, these weren't run
through the seed-variance check — the consistency of the *sign* of the
effect across all four horizons in each strand is suggestive but not
seed-confirmed the way the calendar-feature finding is).
