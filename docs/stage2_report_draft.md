# Stage 2 report content — draft

Draft content for report sections 4–6 (per the assignment's required PDF
structure: improved architecture, improved results, discussion). Written
to be copied into the final report and lightly edited. Sources:
`models/SegRNNTime.py`, `docs/data_pipeline_audit.md`, the Colab run
transcripts.

---

## 4. Improved architecture

**What changed:** `docs/data_pipeline_audit.md` (section 7) found that
`data_provider/data_loader.py` already computes calendar features
(hour-of-day, day-of-week, day-of-month, day-of-year — continuous,
normalized to `[-0.5, 0.5]`) for every timestamp, but `exp/exp_main.py`
discards them on the SegRNN path — only `batch_x` (raw values) is passed
to `model.forward()`, never `batch_x_mark`. `models/SegRNNTime.py` wires
these features into the encoder: each input segment's raw values
(`seg_len` numbers) are concatenated with that segment's calendar feature
vector before the `Linear(w→d)` + ReLU embedding, so `Linear` becomes
`Linear(w + mark_dim → d)` instead of `Linear(w → d)`. Everything else —
GRU encoding, PMF decoding, normalization — is unchanged from
`models/SegRNN.py`.

**Why this should help:** the seasonal-naive baseline (Stage 1) beats
plain naive by a wide margin on ETTh1, confirming a strong daily cycle in
the data. The original SegRNN encoder has no explicit signal for *which
part of that cycle* an input segment belongs to — it must infer this
purely from the recurrent hidden state carried across segments. Giving it
the calendar features directly removes that inference burden.

**A design choice that mattered (worth including in the discussion
section):** the first implementation extracted each segment's calendar
feature vector by *mean-pooling* across the segment's `seg_len=24`
timesteps. This measurably hurt both MSE and MAE relative to the plain
reconstruction (see the "what did not work" note below) rather than
helping. Diagnosis: for hourly ETTh1 with `seg_len=24`, each segment spans
exactly one full day, so mean-pooling `HourOfDay` — the single feature
most likely to carry a useful daily-periodicity signal — averages it to
nearly the same constant for every segment, regardless of which day it
is, destroying exactly the information the change was meant to add. Only
the slower-moving features (day-of-week/month/year) survive pooling
intact, and apparently weren't enough on their own. The fix: take the
segment's **last timestep's** calendar features instead of the mean,
which preserves `HourOfDay` as a real, per-segment discriminating signal.

**Same evaluation protocol:** `scripts/SegRNN/etth1_time.sh` is
byte-for-byte identical to `scripts/SegRNN/etth1.sh` (same `seq_len=720`,
`seg_len=24`, `d_model=512`, dropout, batch size, learning rate, epochs,
patience, seed) — only `--model` differs (plus the new `--mark_dim 4`,
unused by every other model) — so the comparison to the reconstruction
isolates the architecture change.

---

## 5. Improved results

*[PLACEHOLDER — fill in once the corrected (last-timestep) SegRNNTime run
finishes in Colab]*

| Horizon | Paper MSE/MAE | Reconstruction MSE/MAE | Improved MSE/MAE |
|---|---|---|---|
| 96 | 0.351 / 0.392 | 0.3510 / 0.3925 | |
| 192 | 0.392 / 0.414 | 0.3925 / 0.4142 | |
| 336 | 0.423 / 0.433 | 0.4233 / 0.4327 | |
| 720 | 0.466 / 0.472 | 0.4657 / 0.4720 | |

**For the record — the mean-pooling attempt (superseded, not the final
result):** on H=96 and H=192 before the fix, mean-pooled calendar
features gave MSE 0.3548/0.3972 and MAE 0.3956/0.4246 respectively —
consistently *worse* than the reconstruction (+1.1–2.5%), not better.
Kept here because the assignment's discussion section explicitly asks
what did not work, and this is a concrete, diagnosed example rather than
a vague one.

---

## 6. Discussion

*[Draft after the final run — should cover: whether the last-timestep fix
recovered an improvement over the reconstruction; the mean-pooling
failure as a worked example of a plausible-sounding design choice that
turned out to interact badly with a specific hyperparameter (seg_len
equal to the seasonal period); what this suggests about feature
engineering for segment-based models generally (align pooling/extraction
strategy with segment length vs. seasonal period); limitations (only
tested on ETTh1, only the pmf decode path, only channel_id=1).]*
