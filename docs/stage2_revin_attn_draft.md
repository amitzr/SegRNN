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

*[PLACEHOLDER — fill in from notebook Part 8's output]*

| Horizon | Reconstruction MSE | RevIN MSE | Δ | Reconstruction MAE | RevIN MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | | | 0.3925 | | |
| 192 | 0.3925 | | | 0.4142 | | |
| 336 | 0.4233 | | | 0.4327 | | |
| 720 | 0.4657 | | | 0.4720 | | |

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

*[PLACEHOLDER — fill in from notebook Part 9's output]*

| Horizon | Reconstruction MSE | Attention MSE | Δ | Reconstruction MAE | Attention MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | | | 0.3925 | | |
| 192 | 0.3925 | | | 0.4142 | | |
| 336 | 0.4233 | | | 0.4327 | | |
| 720 | 0.4657 | | | 0.4720 | | |

---

## Discussion

*[Draft after both runs. Should cover:]*
- Whether either strand actually beat the reconstruction, and if so by
  how much relative to the seed-to-seed noise floor established in
  `docs/stage2_report_draft.md`'s Part 6 (±0.001–0.005 MSE) — a delta
  smaller than that shouldn't be read as a real effect without its own
  seed check.
- If Attention helped: this would directly validate the
  "bottleneck, not representation" diagnosis from the calendar-feature
  work — strong evidence for *why* those four attempts failed, not just
  *that* they failed.
- If Attention didn't help either: an interesting complementary finding
  to the calendar-feature result — even removing the bottleneck
  structurally doesn't unlock headroom on this dataset/architecture,
  suggesting `h_n`'s single-vector summary was already capturing what
  mattered for ETTh1's relatively short, regular seasonal patterns (as
  opposed to a task with longer or more irregular dependencies where a
  bottleneck would bite harder).
- If RevIN helped: pair with the efficiency sweep's finding as a second
  concrete "free" improvement — worth checking together whether RevIN
  changes which `d_model` is on the efficiency frontier.
- Total Stage 2 picture across all four strands (calendar features,
  efficiency sweep, RevIN, attention): a good spread across "what
  worked, what didn't, what you learned" — from a zero-cost hyperparameter
  flag to the most structurally invasive change attempted, covering both
  representation-level and architecture-level hypotheses.
