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
| 96 | 0.351 | 0.355 | 0.354 | 0.353 | *pending* |
| 192 | 0.392 | 0.397 | 0.396 | 0.400 | *pending* |
| 336 | 0.423 | *n/a* | 0.434 | 0.432 | *pending* |
| 720 | 0.466 | *n/a* | 0.491 | 0.484 | *pending* |

*[Fill in attempt 4's column once the notebook's Part 4 finishes; update
the MAE table the same way. n/a = that attempt was superseded before
running all four horizons.]*

---

## 6. Discussion

*[Draft after attempt 4's run. Should cover:]*
- Whether moving to the decoder (bypassing the bottleneck) finally
  recovered an improvement, or whether even the paper's own
  highest-leverage mechanism can't accommodate this extra signal on
  ETTh1 — in which case the honest conclusion is that SegRNN's PMF
  decoder's positional embedding is already doing what it needs to for
  this dataset, and explicit calendar features are redundant with what
  `rp`/`cp` (learned, data-driven) already capture implicitly.
- The four-attempt arc as a worked example of debugging a negative
  result properly: first ruling out an implementation bug (pooling vs.
  seasonal period), then a representational-capacity issue (raw scalar
  vs. embedding, directly informed by course material), then an
  architectural-placement issue (encoder bottleneck vs. decoder
  side-channel, also informed by course material) — three genuinely
  different hypotheses, each tested cleanly in isolation.
- What this suggests generally: for an architecture that already has a
  demonstrated high-leverage side-channel (PE), adding new information
  there is a better first move than adding it to the main representation
  path, regardless of how well-represented the new information is.
- Limitations: only tested on ETTh1, only the `pmf` decode path, only
  `channel_id=1`, embedding dimensions not swept.
