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
to `model.forward()`, never `batch_x_mark`. `models/SegRNNTime.py` wires
these features into the encoder. The final design: each input segment's
raw values (`seg_len` numbers) are concatenated with (a) a learned
embedding of that segment's hour-of-day, looked up via
`nn.Embedding(24, hour_emb_dim)`, and (b) its remaining calendar features
(day-of-week/month/year) as raw scalars — before the
`Linear(seg_len + hour_emb_dim + 3 → d_model)` + ReLU embedding.
Everything else — GRU encoding, PMF decoding, normalization — is
unchanged from `models/SegRNN.py`.

**Why this should help:** the seasonal-naive baseline (Stage 1) beats
plain naive by a wide margin on ETTh1, confirming a strong daily cycle in
the data. The original SegRNN encoder has no explicit signal for *which
part of that cycle* an input segment belongs to — it must infer this
purely from the recurrent hidden state carried across segments. Giving it
the calendar features directly removes that inference burden.

**Why an embedding, specifically — grounded in the course material:**
`docs/DL_for_TS.pdf`'s "Include the year and month as predictor
variables" → "Improved Model" → "Embedding Layer" sequence teaches exactly
this distinction: route a cyclical/categorical calendar feature (there,
month — 12 categories) through a learned `Embedding`, and only treat
genuinely linear quantities (there, year) as raw scalars. A single raw
continuous number forces the network to learn one smooth function of that
number to represent 24 (or 12, or 7) qualitatively different categories;
an embedding gives each category its own independent learned vector.

**Two earlier, less-grounded attempts (both negative — see section 5):**
before landing on the embedding, two versions concatenated `HourOfDay` as
a single raw continuous scalar instead:
1. **Mean-pooled** across each segment's `seg_len=24` timesteps. Since
   `seg_len=24` for hourly ETTh1 means each segment spans exactly one full
   day, mean-pooling `HourOfDay` — the feature most likely to carry a
   useful daily-periodicity signal — averages it to nearly the same
   constant for every segment, destroying exactly the information the
   change was meant to add.
2. **Last-timestep** (fixes the pooling-cancellation bug above, but still
   only a single continuous number for a 24-way categorical quantity).

**Same evaluation protocol:** all three attempts use identical
hyperparameters, split, and seed as the reconstruction (`seq_len=720`,
`seg_len=24`, `d_model=512`, dropout, batch size, learning rate, epochs,
patience) — only the model and its calendar-feature-specific flags
(`--mark_dim`, `--hour_emb_dim`) differ, so the comparison isolates the
architecture change.

---

## 5. Improved results

Paper vs. reconstruction (both from Stage 1, unchanged):

| Horizon | Paper MSE/MAE | Reconstruction MSE/MAE |
|---|---|---|
| 96 | 0.351 / 0.392 | 0.3510 / 0.3925 |
| 192 | 0.392 / 0.414 | 0.3925 / 0.4142 |
| 336 | 0.423 / 0.433 | 0.4233 / 0.4327 |
| 720 | 0.466 / 0.472 | 0.4657 / 0.4720 |

**Attempt 1 — mean-pooled raw scalar** (superseded): MSE/MAE
0.3548/0.3956 (H=96), 0.3972/0.4246 (H=192) — consistently *worse* than
the reconstruction (+1.1–2.5%).

**Attempt 2 — last-timestep raw scalar** (superseded): confirmed via a
full, bug-free run of all four horizons:

| Horizon | Reconstruction | Attempt 2 (last-timestep raw) | Change |
|---|---|---|---|
| 96 | 0.351 / 0.393 | 0.354 / 0.395 | +0.9% MSE |
| 192 | 0.392 / 0.414 | 0.396 / 0.424 | +1.0% MSE |
| 336 | 0.423 / 0.433 | 0.434 / 0.441 | +2.6% MSE |
| 720 | 0.466 / 0.472 | 0.491 / 0.490 | +5.4% MSE |

Still consistently worse, and the gap *grows* with horizon (more segments
in the encoder → more opportunities for a weakly-integrated raw scalar to
act as noise rather than signal).

**Attempt 3 — hour-of-day embedding (current design):**

*[PLACEHOLDER — fill in once `models/SegRNNTime.py`'s embedding version
finishes running in the notebook's Part 4]*

| Horizon | Paper MSE/MAE | Reconstruction MSE/MAE | Improved (embedding) MSE/MAE |
|---|---|---|---|
| 96 | 0.351 / 0.392 | 0.3510 / 0.3925 | |
| 192 | 0.392 / 0.414 | 0.3925 / 0.4142 | |
| 336 | 0.423 / 0.433 | 0.4233 / 0.4327 | |
| 720 | 0.466 / 0.472 | 0.4657 / 0.4720 | |

---

## 6. Discussion

*[Draft after the final run. Should cover:]*
- Whether the embedding attempt recovered an actual improvement over the
  reconstruction, or whether even a properly-grounded technique still
  didn't beat a model this well-tuned by the original authors.
- The two earlier failures as a worked example: a plausible-sounding
  feature-engineering idea (concatenate calendar features) can fail for
  reasons that have nothing to do with the idea itself — first an
  interaction between pooling strategy and `seg_len` equal to the seasonal
  period, then a representational-capacity issue (raw scalar vs.
  embedding) that the course's own lecture material had already flagged
  the fix for.
- What this suggests about feature engineering for segment-based models
  generally: alignment between pooling/extraction strategy and both the
  segment length *and* the feature's cardinality/cyclicality matters as
  much as whether the feature is informative in principle.
- Limitations: only tested on ETTh1, only the `pmf` decode path, only
  `channel_id=1`, only `hour_emb_dim=16` (not swept).
