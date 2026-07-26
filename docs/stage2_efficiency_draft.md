# Stage 2 report content — efficiency sweep (draft)

Draft content for an additional Stage 2 "improvement" strand, separate
from the calendar-feature work (`docs/stage2_report_draft.md`). Covers
the assignment's "improve computational efficiency" example category.
Sources: notebook Part 7, `results/runs.csv`.

---

## Improved architecture (efficiency variant)

**What changed:** nothing in the code — `--d_model` is already a CLI
flag on the unmodified `run_longExp.py`/`models/SegRNN.py`. This isn't an
architecture change, it's a hyperparameter sweep: the reconstruction
(`d_model=512`, the paper's default) rerun at `d_model ∈ {256, 128, 64}`
across all four horizons, holding every other hyperparameter fixed
(`seq_len=720`, `seg_len=24`, dropout, batch size, learning rate, epochs,
patience, seed=2024).

**Why this is a reasonable improvement candidate:** `d_model` sets the
width of the segment embedding, the GRU hidden state, and the PMF
decoder's hidden state — it's the dominant factor in both parameter count
and per-step compute. The paper doesn't report a hidden-size ablation
for SegRNN specifically (only PatchTST's default width in the runtime
comparison, Table VI), so this explores an axis the paper doesn't cover
for this model. Unlike the calendar-feature work, this doesn't rely on
any hypothesis about what information the model needs — it's a direct,
guaranteed-to-produce-a-result tradeoff study between accuracy and
inference cost.

---

## Results

*[PLACEHOLDER — fill in from the notebook's Part 7 output:
`results/figures/efficiency_sweep.png` and the two pivot tables (MSE,
ms/sample) by horizon × d_model]*

MSE by horizon × `d_model`:

| Horizon | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| 96 | | | | |
| 192 | | | | |
| 336 | | | | |
| 720 | | | | |

Inference cost (ms/sample) by horizon × `d_model`:

| Horizon | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| 96 | | | | |
| 192 | | | | |
| 336 | | | | |
| 720 | | | | |

**Reading the curve:** *[fill in — where does accuracy start degrading
noticeably as d_model shrinks? Is the accuracy-cost tradeoff roughly
linear, or is there a "knee" where a smaller d_model gives most of the
speedup with little accuracy cost? Does the answer differ by horizon —
e.g. does a longer horizon need more capacity (bigger d_model) to hold
up, while a short horizon tolerates a much smaller one?]*

---

## Discussion

*[Draft after the run. Should cover:]*
- The headline number: at the best tradeoff point found, X% of the
  d_model=512 accuracy retained at Y% of the inference cost (or params).
- Whether this generalizes across all four horizons or is horizon-specific.
- How this complements the calendar-feature finding: two genuinely
  different kinds of Stage 2 result — one that improved a real,
  practical axis (efficiency) the paper didn't explore for this model,
  one that tested a specific hypothesis about missing information and
  found, with seed-confirmed rigor, that it doesn't help. Together they
  cover more of "what worked, what didn't, what you learned" than either
  alone.
- Practical framing: SegRNN's own pitch (per the paper) is efficiency
  relative to Transformer baselines (>78% faster, >82% less memory than
  PatchTST) — this sweep asks whether SegRNN itself has further headroom
  on the same axis, a natural extension of the paper's own stated
  motivation.
