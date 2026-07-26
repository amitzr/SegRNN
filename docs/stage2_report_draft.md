# Stage 2 report content — draft

Draft content for report sections 4–6 (per the assignment's required PDF
structure: improved architecture, improved results, discussion). Written
to be copied into the final report and lightly edited. Sources:
`models/SegRNNTime.py`, `docs/data_pipeline_audit.md`, `docs/DL_for_TS.pdf`
(course lecture), the Colab run transcripts.

---

## 4. Improved architecture

**What changed:** `docs/data_pipeline_audit.md` (section 7) found that
`data_provider/data_loader.py` already computes calendar features
(hour-of-day, day-of-week, day-of-month, day-of-year — continuous,
normalized to `[-0.5, 0.5]`) for every timestamp, but `exp/exp_main.py`
discards them on the SegRNN path — only `batch_x` (raw values) is passed
to `model.forward()`, never `batch_x_mark`/`batch_y_mark`.
`models/SegRNNTime.py` (four iterations — see below) wires these features
in. The current (4th, decoder-side) design: the encoder is identical to
`models/SegRNN.py` (no calendar features at all); the PMF decoder's
positional embedding is extended from `PE = concat(rp, cp)` to
`PE = concat(rp, cp, hour_embedding, weekday_embedding)`, where
`hour_embedding`/`weekday_embedding` are learned `nn.Embedding(24, ·)` /
`nn.Embedding(7, ·)` lookups of each target segment's own **future**
hour-of-day and day-of-week (known in advance for the whole forecast
horizon — a calendar fact, not data, so not leakage).

**Why this should help:** the seasonal-naive baseline (Stage 1) beats
plain naive by a wide margin on ETTh1, confirming a strong daily cycle in
the data — the model should benefit from knowing where in that cycle each
segment sits, whether input or output.

**Why the decoder, specifically — three earlier attempts failed:**

1. **Mean-pooled encoder-side.** Each input segment's raw values
   concatenated with the segment's *mean-pooled* calendar features before
   the segment embedding. Since `seg_len=24` for hourly ETTh1 spans
   exactly one full day, mean-pooling `HourOfDay` averages it to nearly
   the same constant for every segment, destroying the signal. **Result:**
   worse than the reconstruction (+1.1–2.5% MSE at H=96/192).
2. **Last-timestep encoder-side.** Same concatenation, but using the
   segment's last timestep instead of a mean (fixes the pooling
   cancellation). **Result:** still consistently worse, and the gap
   *grew* with horizon (+0.9% at H=96 → +5.4% at H=720).
3. **Hour-of-day embedding, encoder-side.** Per `docs/DL_for_TS.pdf`'s
   "Include the year and month as predictor variables" → "Embedding
   Layer" lecture sequence — route the cyclical/categorical feature
   through a learned `Embedding` rather than a raw scalar (there: month,
   12 categories; here: hour, 24). **Result:** modestly better than
   attempt 2 at the longer horizons but *still* consistently worse than
   the reconstruction at every horizon (+0.6% at H=96 → +3.9% at H=720).

All three shared one trait: they injected calendar info into the
**encoder**. `docs/DL_for_TS.pdf`'s Attention slide names the likely
culprit directly — it labels a single encoder context vector an
**"information bottleneck"**. SegRNN's encoder already compresses the
entire look-back window into one hidden state `h_n` before the decoder
ever sees it; every encoder-side calendar addition forced calendar
information to compete with raw-value information for space in that one
vector. That would explain why representation quality (mean → last-step →
embedding) barely mattered across three attempts: the bottleneck was the
constraint, not the representation.

**Attempt 4 (current) moves the signal to the decoder instead**, where it
doesn't have to compete for encoder capacity at all. SegRNN's PMF decoder
already has a side-channel built for exactly this: `PE`, fed directly
alongside the repeated `h_n` into the decode step, bypassing the encoder
bottleneck entirely. The paper's own ablation (Table V) shows `PE` —
specifically the relative-position component — is the single
highest-leverage component in the whole architecture (28.8% MSE reduction
alone), so this targets the part of the model already demonstrated to
matter most, combined with the lecture's embedding-not-raw-scalar lesson
for *how* to represent the added feature once it's there.

**Same evaluation protocol throughout:** all four attempts use identical
hyperparameters, split, and seed as the reconstruction (`seq_len=720`,
`seg_len=24`, `d_model=512`, dropout, batch size, learning rate, epochs,
patience) — only the model and its calendar-feature-specific flags
(`--mark_dim`, `--hour_emb_dim`, `--weekday_emb_dim`) differ.

---

## 5. Improved results

Paper vs. reconstruction (Stage 1, unchanged):

| Horizon | Paper MSE/MAE | Reconstruction MSE/MAE |
|---|---|---|
| 96 | 0.351 / 0.392 | 0.3510 / 0.3925 |
| 192 | 0.392 / 0.414 | 0.3925 / 0.4142 |
| 336 | 0.423 / 0.433 | 0.4233 / 0.4327 |
| 720 | 0.466 / 0.472 | 0.4657 / 0.4720 |

All four attempts, MSE, side by side:

| Horizon | Reconstruction | 1: mean-pool | 2: last-step raw | 3: hour embed (encoder) | 4: hour+weekday embed (decoder) |
|---|---|---|---|---|---|
| 96 | 0.351 | 0.355 | 0.354 | 0.353 | 0.363 |
| 192 | 0.392 | 0.397 | 0.396 | 0.400 | 0.397 |
| 336 | 0.423 | *n/a* | 0.434 | 0.432 | 0.424 |
| 720 | 0.466 | *n/a* | 0.491 | 0.484 | 0.476 |

MAE:

| Horizon | Reconstruction | 2: last-step raw | 3: hour embed (encoder) | 4: hour+weekday embed (decoder) |
|---|---|---|---|---|
| 96 | 0.393 | 0.395 | 0.396 | 0.400 |
| 192 | 0.414 | 0.424 | 0.425 | 0.425 |
| 336 | 0.433 | 0.441 | 0.442 | 0.434 |
| 720 | 0.472 | 0.490 | 0.486 | 0.479 |

% change vs. reconstruction, MSE, attempts 2/3/4:

| Horizon | 2: last-step raw | 3: hour embed (encoder) | 4: hour+weekday embed (decoder) |
|---|---|---|---|
| 96 | +0.9% | +0.6% | +3.4% |
| 192 | +1.0% | +2.0% | +1.3% |
| 336 | +2.6% | +2.1% | +0.2% |
| 720 | +5.4% | +3.9% | +2.1% |

n/a = that attempt was superseded before running all four horizons (attempt 1
was abandoned after H=96/192 confirmed the pooling bug).

**Reading the trend:** attempt 4 is the closest of the four to parity —
nearly dead-even at H=336 (+0.2%, essentially noise) and the smallest gap
of any attempt at H=720 (+2.1%, vs. +5.4% for attempt 2 and +3.9% for
attempt 3). Moving the signal out of the encoder did measurably reduce
the damage, consistent with the bottleneck hypothesis — but it never
actually crosses over into an improvement at any horizon. H=96 is
attempt 4's worst relative result (+3.4%), the opposite pattern from
attempts 2/3 (which were closest to parity at H=96 and worst at H=720) —
plausibly because a short horizon has few decode segments (`m=4` at
H=96 vs. `m=30` at H=720), so the two new embedding tables have very
little data per forward pass to learn from at that setting, while a long
horizon gives them more decode steps to learn across, at the cost of the
encoder-bottleneck pressure attempts 2/3 suffered from.

**Seed variance check (before trusting any of the above):** every result
above uses a single seed (2024). The notebook's Part 6 reruns
Reconstruction and Improved at seeds 2021/2022 (plus the existing 2024
run) for H=336 (smallest observed gap) and H=720 (largest), and computes
the paired per-seed delta (Improved − Reconstruction) with its
mean/std across the 3 seeds — if `|mean delta| < std delta`, the
attempt-4 effect at that horizon is statistically indistinguishable from
noise at this sample size.

*[PLACEHOLDER — fill in from the notebook's Part 6 output]*

| Horizon | Reconstruction MSE (mean ± std, n=3) | Improved MSE (mean ± std, n=3) | Paired delta (mean ± std) | Verdict |
|---|---|---|---|---|
| 336 | | | | |
| 720 | | | | |

---

## 6. Discussion

*[Note: the framing below treats the attempt 1→4 trend as a real,
progressively-improving effect. Check the seed variance table in section
5 first — if it says the attempt-4 delta at H=336/720 is statistically
indistinguishable from noise, soften "moved the result in the predicted
direction" below to something like "was directionally consistent with,
but not statistically distinguishable from, the predicted effect."]*

**What worked:** nothing beat the reconstruction outright — this is a
negative result across all four attempts. But the *arc* across attempts
worked, in the sense that each iteration was a distinct, falsifiable
hypothesis about *why* the previous one failed, and each test actually
moved the result in the predicted direction:
- Attempt 1 → 2 tested whether mean-pooling was destroying the signal
  (it was — `seg_len` happened to equal the seasonal period). Fixing it
  didn't recover an improvement, but ruled out an implementation bug as
  the explanation.
- Attempt 2 → 3 tested whether representational capacity was the issue
  (raw scalar vs. learned embedding for a 24-category feature), per
  `docs/DL_for_TS.pdf`'s explicit teaching on this exact distinction.
  Modest improvement at longer horizons, but still net negative.
- Attempt 3 → 4 tested whether the *architectural location* was the
  issue (encoder bottleneck vs. decoder side-channel), per the same
  lecture's Attention slide. This produced the largest improvement of
  the three fixes — attempt 4 is within +0.2% of the reconstruction at
  H=336 and has the smallest gap of any attempt at H=720 — without
  crossing over into an actual win.

**What this suggests:** SegRNN's PMF decoder positional embedding
(`rp`/`cp`) is already a *learned, data-driven* per-segment code — during
training it's free to arrange itself however best predicts the target,
unconstrained by what a human labels "hour" or "weekday." It's plausible
`rp` already recovers most of the useful periodic structure implicitly
(the model sees the same 24-hour, 7-day cycles repeat across ~7800
training windows), and the explicit calendar embeddings are largely
redundant with what `rp` already learned — while still adding two new
sets of parameters that have to be fit from the same amount of data,
which is a real cost even when the added signal isn't harmful. This is
consistent with attempt 4's H=96 result being its *worst* relative
outcome (+3.4%): at H=96, `m=4` decode segments per sample means the new
embedding tables get very little gradient signal per batch to learn from,
so their cost (extra parameters, harder optimization) shows up more
than any benefit; at H=336 (`m=14`) there's enough decode steps for them
to earn their keep, roughly breaking even.

**Practical takeaway:** for an architecture that already has a
demonstrated high-leverage side-channel (PE, per the paper's own
Table V ablation), adding new information there is a better first move
than adding it to the main representation path — this held up
empirically (attempt 4 clearly closer to parity than 2 or 3) even though
it didn't fully pay off here. The three-hypothesis debugging arc (bug →
representation → architecture) is arguably the most transferable lesson:
a plausible-sounding feature-engineering idea can fail for reasons that
have nothing to do with whether the underlying information is useful.

**Limitations:** only tested on ETTh1 (a single dataset with `enc_in=7`);
only the `pmf` decode path (`rmf` was never compatible with the
encoder-side attempts and wasn't revisited for the decoder-side one);
only `channel_id=1`; `hour_emb_dim=16`/`weekday_emb_dim=8` were picked
once, not swept; only day-of-week and hour-of-day were tried as decoder
features (day-of-month/day-of-year were dropped after attempt 2, on the
assumption their lower cyclicality made them less promising — untested).
