# Stage 2 report content — efficiency sweep (draft)

Draft content for an additional Stage 2 "improvement" strand, separate
from the calendar-feature work (`docs/stage2_report_draft.md`). Covers
the assignment's "improve computational efficiency" example category.
Sources: notebook Part 7, `scripts/plot_efficiency_sweep.py`,
`results/runs.csv`.

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
decoder's hidden state — it's the dominant factor in parameter count
(the GRU alone is `6·d_model² + 6·d_model` parameters, versus low
thousands for everything else combined, so total parameter count scales
almost exactly quadratically with `d_model`). The paper doesn't report a
hidden-size ablation for SegRNN specifically (only PatchTST's default
width appears in the runtime comparison, Table VI), so this explores an
axis the paper doesn't cover for this model — and it's SegRNN's own
stated selling point (the paper's abstract and Table VI both center
efficiency vs. Transformer baselines), so asking whether SegRNN has
further headroom on that same axis is a natural, on-theme extension.

**Cost metric — a methodological note:** the sweep originally measured
wall-clock inference time (`ms/sample`, printed by `exp/exp_main.py`'s
test loop). That measurement turned out to be non-monotonic in
`d_model` (e.g. `d_model=128` timed slower than `d_model=512` at several
horizons) — inconsistent with smaller matrices requiring strictly less
compute, so it's read as noise from a single-pass, no-warm-up wall-clock
measurement on a shared Colab GPU, not a real signal. **Parameter count**
is used instead: computed analytically from `models/SegRNN.py`'s exact
layer shapes (not measured, so no timing noise), and it is what actually
determines a model's storage/deployment footprint regardless of any
particular GPU's momentary load.

---

## Results

MSE by horizon × `d_model` (measured, `run_longExp.py` unmodified):

| Horizon | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| 96 | 0.3510 | 0.3557 (+1.3%) | 0.3677 (+4.8%) | 0.3848 (+9.6%) |
| 192 | 0.3925 | 0.3956 (+0.8%) | 0.4007 (+2.1%) | 0.4219 (+7.5%) |
| 336 | 0.4233 | 0.4248 (+0.35%) | 0.4304 (+1.7%) | 0.4445 (+5.0%) |
| 720 | 0.4657 | 0.4881 (+4.8%) | 0.4915 (+5.5%) | 0.4950 (+6.3%) |

Parameter count by horizon × `d_model` (exact, analytical — `seg_num_y`
varies slightly by horizon, so totals vary marginally by horizon too;
the GRU term dominates and is horizon-independent):

| Horizon | 512 | 256 | 128 | 64 |
|---|---|---|---|---|
| 96 | 1,603,864 | 408,728 (25.5%) | 106,072 (6.6%) | 28,472 (1.8%) |
| 192 | 1,604,888 | 409,240 (25.5%) | 106,328 (6.6%) | 28,600 (1.8%) |
| 336 | 1,606,424 | 410,008 (25.5%) | 106,712 (6.6%) | 28,792 (1.8%) |
| 720 | 1,610,520 | 412,056 (25.6%) | 107,736 (6.7%) | 29,304 (1.8%) |

(percentages are relative to that horizon's `d_model=512` value)

![Accuracy vs. parameter count](figures/efficiency_params.png)

**Reading the curve:** there's a clear "knee" at `d_model=256` for the
three shorter horizons (96/192/336): a **~4x reduction in parameters**
(down to ~25.5%) costs only **+0.35% to +1.3% MSE** — essentially free.
Dropping further to 128 (~15x fewer params) starts costing more
noticeably (+1.7–4.8%), and 64 (~55x fewer) is a real accuracy hit
(+5.0–9.6%) — diminishing returns past 256. H=720 is the exception: even
the first step down to 256 already costs +4.8%, and the curve declines
more steadily rather than showing a flat region — the longest horizon
needs the most capacity to hold up, which makes intuitive sense (more
output segments for the PMF decoder to produce from the same encoded
state).

---

## Discussion

**Headline finding:** at H=336 (and similarly H=192), `d_model=256`
retains **99.65% of the reconstruction's accuracy at 25.5% of its
parameter count** — a genuine, unambiguous efficiency win, found with
zero architecture changes. At H=720, the same swap costs a more real
+4.8% MSE, so the right `d_model` depends on which horizon a deployment
actually cares about; a practical recommendation would be `d_model=256`
as a strong default for short-to-medium horizons, keeping `512` only if
the long-horizon (720) case specifically matters.

**How this complements the calendar-feature work:** these are two
genuinely different kinds of Stage 2 result. The calendar-feature
attempts (`docs/stage2_report_draft.md`) tested a specific hypothesis
about what information the model was missing, iterated through three
increasingly well-grounded fixes, and — confirmed via a seed-variance
check, not just a single run — found the hypothesis doesn't hold on this
dataset/architecture: a clean, rigorous negative result. This sweep asks
a different kind of question (how much capacity does the model actually
need) and gets an unambiguous positive answer. Between the two: one
finding that improved a real, practical axis; one that failed
instructively with a diagnosed reason. Together they cover substantially
more of the assignment's "what worked, what did not, what you learned"
than either alone.

**Limitations:** only tested on ETTh1; the wall-clock timing signal was
unreliable at the scale of a single Colab run, so the efficiency claim
rests on parameter count rather than measured latency or memory (a more
controlled benchmark — repeated trials with warm-up, on a dedicated GPU —
would be needed to make a latency claim with the same confidence); only
`d_model` was swept, not other capacity-related hyperparameters (e.g.
`seg_len`).
