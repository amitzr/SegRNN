# Stage 2 — full strand inventory, for planning the combination experiment

Every Stage 2 strand tried so far, in one table, organized by the
instructor's own improvement categories (`docs/segrnn_project_plan.md`'s
"Coverage of the instructor's improvement categories" table — anomaly/
change-point detection and probabilistic forecasting/uncertainty are
omitted, neither was pursued). "In lectures?" checks against all three
uploaded decks (`docs/DL_for_TS.pdf`, `docs/Pre-precessing.pdf`,
`docs/Time-Series Forecasting.pdf`) — Yes/Partial/No, not just whether
the general topic came up somewhere. Sources: `docs/stage2_report_draft.md`,
`docs/stage2_efficiency_draft.md`, `docs/stage2_revin_attn_draft.md`,
`docs/stage2_new_strands_draft.md`, `docs/stage2_architecture_variants_draft.md`,
`docs/stage2_followup_ideas_draft.md`. Written before deciding which
combinations to build or test — this is the input to that decision, not
the decision itself.

---

## 1. Improve preprocessing and data quality

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| RevIN, `affine=False` | Regressed at every horizon, +3.1% to +4.6% MSE | Yes (`DL_for_TS.pdf`, normalization slide) | **No** |
| RevIN, `affine=True` | Better than `affine=False`, still net worse than no RevIN | Yes (same slide; this variant is this project's own follow-up, not itself lecture content) | **No** |
| Yeo-Johnson power transform | MSE **-3.6% to -7.4%** at every horizon; MAE **+4.1% to +6.7%** at every horizon — a real trade-off, not a clean win | Yes (`Pre-precessing.pdf`) | **Yes** — best single MSE result in the project; combine only where MSE is the metric that matters, and check what it does to MAE in any combination |
| Box-Cox (MLE-fit and fixed-lambda sweep) | **Dropped, not measured** — built on the assumption ETTh1 is strictly positive (Box-Cox's requirement); sklearn's own check proved that false, and a related implementation bug (no positivity validation in the fixed-lambda path) silently produced a wasted NaN training run before the false premise was caught | Yes (`Pre-precessing.pdf`, as the positive-only sibling of Yeo-Johnson) | **N/A** — abandoned before any usable result existed, not a "no," a non-result |
| Yeo-Johnson, fixed-lambda sweep (one global λ, `{-2,...,2}`) | No fixed λ beats the MLE-fit MSE. But **λ=-2 gives MAE=0.2832 at H=336 — the best MAE anywhere in this project** — at the cost of MSE nearly double the reconstruction (0.78 vs. 0.42). λ=1 exactly reproduces the reconstruction (mathematical identity), confirming the implementation | Yes (`Pre-precessing.pdf`; the sweep methodology itself, not a specific λ, is the tested idea) | **No for combination as-is** — confounded with per-channel vs. global λ (MLE fits 7 channel-specific λ, this sweep forces one shared λ), so it doesn't cleanly settle MLE-vs-not; the λ=-2 MAE number is a genuine extreme worth citing, not a deployable config on its own |

## 2. Add stronger time-series features

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| Calendar features in positional embedding (3 attempts: encoder x2, decoder) | Regressed at every horizon, every attempt, seed-confirmed real | Partial (`DL_for_TS.pdf` embedding-layer slide covers embedding categorical/cyclical features generally, not this specific use) | **No** |
| ROCKET random-convolution features | Regressed at 3/4 horizons; H=336's +8.7% MSE is the worst single-horizon regression in the project | Yes, as a technique (`Pre-precessing.pdf`), but classification-era, adapted to regression without precedent | **No** |
| Top-K FFT-magnitude features (with and without Yeo-Johnson) | Qualitatively "terrible" (exact table not yet transcribed) | Partial (`Pre-precessing.pdf` covers window/frequency features generally, not top-K FFT specifically) | **No** |
| Multi-scale window statistics (mean/std/slope, day/week/full-lookback) | Roughly flat at H=96, real small win at H=192/336, **sharp regression at H=720** (+9.2% MSE, +8.4% MAE — the worst single per-horizon number in the project) | Yes (`Pre-precessing.pdf`, "window-based features" is its own named taxonomy item) | **Maybe** — only for short-to-medium horizon deployments; avoid at H=720. Not seed-confirmed, unlike most other headline results here |

## 3. Improve or tune the forecasting model

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| AIC/BIC post-hoc model-selection analysis | Not a trainable change — free re-analysis of the `d_model` sweep. Both criteria reject `d_model=512` at every horizon, favor 64/128 over even 256 | Yes (`Time-Series Forecasting.pdf`) | **Use for justification**, not as a component — it's analysis, not something to "add" to a model |
| Multi-seed prediction ensembling | Borderline noise at H=336, clearly noise at H=720 — no real effect either way | No (flagged honestly as a general technique, not from these lectures) | **Optional** — cheap wrapper, doesn't preclude combining with anything else, but don't expect it to move results |
| Huber loss (alone) | **H=720: MSE -4.4%, MAE -3.3% — both improve, no trade-off.** H=336: small, balanced (+0.9% MSE, -0.5% MAE) | No (general ML technique; motivated by this project's own Yeo-Johnson finding, not a lecture citation) | **Yes** — strongest, cleanest win in the whole project |
| Huber loss + Yeo-Johnson | H=720: MSE barely changes (-4.5%) but MAE flips to **+3.1%** — strictly worse than Huber alone | No | **No** — the two compete for the same job, don't stack them |
| Blended MSE+MAE loss (alone) | Smaller version of Huber's pattern: H=720 both metrics improve (-1.3%/-1.2%), H=336 small and balanced | No | **Maybe** — same direction as Huber, weaker; Huber is the better pick of the two |
| Blended loss + Yeo-Johnson | Same competing-fixes pattern as Huber+YJ | No | **No** |
| Post-hoc `SegRNN`+`DLinear` prediction ensemble | Small, mixed: MAE better at 3/4 horizons, MSE better at H=720 (-2.2%), MAE slightly worse at H=720 (+0.8%) | No (paper-motivated — DLinear's own competitiveness — not lecture-motivated) | **Maybe** — real but modest; adds deployment complexity (two trained models) for a small gain |

## 4. Improve ML or deep-learning architecture

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| Attention over encoder states (`SegRNNAttn`) | Regressed at every horizon, +0.5% to +5.1% MSE | Yes (`DL_for_TS.pdf`, information-bottleneck diagnosis) | **No** |
| Encoder depth (stacked GRU, `num_layers=2`) | Slightly worse | No | **No** |
| Encoder direction (bidirectional + forced separate decode cell) | Roughly neutral — a surprise given it's the heaviest strand by parameter count | No | **No** — neutral at real added cost isn't worth it |
| Value embedding via `Conv1d` (narrower than segment) | Bad | Partial (`Pre-precessing.pdf` covers convolutional/window features generally) | **No** |
| Un-shared decode cell (unidirectional encoder) | Regresses, worse as horizon grows | No | **No** |
| Parameter-free pooling context (mean-pool encoder states into `h_n`) | **MSE and MAE both improve at 3/4 horizons (96, 192, 720); H=336 a wash. Zero added parameters.** | Partial (motivated by `DL_for_TS.pdf`'s bottleneck diagnosis; pooling itself isn't the lecture's prescribed fix — attention is) | **Yes** — cleanest, cheapest win in the entire project |
| Weight-tied value-embedding/predict weights | Mixed/inconclusive: MSE slightly worse, MAE slightly better only at H=720, rest ~unchanged | No | **Maybe** — weak evidence either way, low risk to include if efficiency (not accuracy) is the goal |
| Linear shortcut, jointly-trained blend gate | Worse (superseded by the post-hoc ensemble above) | No | **No** |
| LayerNorm after input embedding | Worse (superseded by the `h_n` version below) | No | **No** |
| LayerNorm on `h_n` (hidden state, before decode) | Worse at H=96/192/336; **clean win on both metrics at H=720** (-4.7% MSE, -2.7% MAE) | No | **Maybe** — only if the deployment horizon is long; a regression at shorter horizons otherwise |
| Frozen recurrent cell, naive init (`SegRNNReservoir`, no ESN tuning) | Regressed (user-reported; exact per-horizon table not yet transcribed) | No (ESN/reservoir computing appears nowhere in the paper or these lectures) | **No** |
| Frozen recurrent cell, proper ESN init (spectral radius 0.9) | Regressed (user-reported; exact per-horizon table not yet transcribed) | No | **No** |
| Frozen recurrent cell, spectral radius sweep `{0.5,0.9,0.99,1.1}` | Regressed across the sweep (user-reported; exact table not yet transcribed) | No | **No** — worth checking whether the sweep at least shows the predicted radius-1 degradation before writing this off as fully uninformative |

## 5. Improve computational efficiency

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| `d_model` sweep (512 -> 256/128/64) | **`d_model=256` keeps 99.65% of accuracy at 25.5% of the parameters, H<=336.** H=720 costs more (+4.8%). AIC/BIC (category 3 above) independently argues for going even smaller (64/128) | Not explicitly when first run; reinforced after the fact by AIC/BIC (`Time-Series Forecasting.pdf`) | **Yes** — the project's clearest efficiency result, and the one every capacity-adding strand above has been implicitly tested against |

---

## Quick reference: strongest candidates for the combination experiment

Everything marked **Yes** above, pulled into one place:

1. **`d_model=256`** (efficiency)
2. **Yeo-Johnson power transform** (preprocessing) — MSE win, MAE cost
3. **Huber loss** (training objective) — the cleanest win in the project, but conflicts with Yeo-Johnson (don't combine those two)
4. **Parameter-free pooling context** (architecture) — zero cost, the other clean win

Everything marked **Maybe** is a secondary/optional layer on top of those
four, not a starting point: blended loss (weaker version of Huber),
weight tying (marginal), `h_n` LayerNorm (long-horizon only),
`SegRNN`+`DLinear` ensembling (small gain, real added complexity),
multi-seed ensembling (cheap, no expected effect but no expected harm).

Not yet addressed here: which of these four actually *compose* — e.g.
Huber and pooling context both touch different parts of the pipeline
(loss vs. architecture) and have no obvious conflict, but Huber and
Yeo-Johnson are already known to compete for the same job. That
compatibility analysis is the next step, before writing any combination
code.

---

## Combination plan: the seven strands going into the final report

The seven chosen for presentation: (1) Yeo-Johnson, MLE-fit and the
fixed-lambda sweep; (2) multi-scale window statistics; (3) AIC/BIC
analysis, then `d_model`; (4) Huber loss; (5) encoder direction
(bidirectional); (6) pooling context; (7) the `DLinear` post-hoc
ensemble.

**One correction before building anything:** strand 6 was tested with
**mean** pooling (`--pool_type mean`, the default) — `SegRNNPoolContext.py`
supports max pooling too, but it was never actually run. The "both
metrics improve at 3/4 horizons" result on record is for mean pooling.
If the report specifically wants to say "max pool," that result doesn't
exist yet and would need its own run first.

### Two different kinds of combination — this matters for cost, not just logic

These seven strands sit at different levels of the pipeline, and that
changes what "combining" costs, not just whether it's logically sound:

**Free — no new code, just multiple flags on one run.** Yeo-Johnson
(`--power_transform`/`--yj_lambda`), Huber loss (`--loss huber`), and
`d_model` are all handled in `data_provider`/`exp_main.py`'s training
loop, completely independent of which model class is selected. Any of
`SegRNN`, `SegRNNWindowStats`, `SegRNNBidir`, `SegRNNPoolContext` can
already be run with any mix of these three today —
`run_horizon('SegRNNPoolContext', h, power_transform=4, yj_lambda=-2,
loss='huber', d_model=256)` is a real, already-supported call. No new
model file needed for any combination that only touches these three
axes.

**Needs new code — two architecture ideas living in two separate model
classes.** Window stats, encoder direction, and pool context are each
their own file (`SegRNNWindowStats.py`, `SegRNNBidir.py`,
`SegRNNPoolContext.py`). Combining any two of *these* with each other
(e.g., pool context *and* window stats in the same forward pass) means
writing a new hybrid class — only one `--model` can be selected per run.
The `DLinear` ensemble is a third case: a notebook-level operation
(train two full models, average predictions afterward), composable with
anything used to train the `SegRNN` side, but see the gotcha below
before mixing it with a preprocessing change.

### The one known conflict — don't repeat it

Huber loss and Yeo-Johnson already went head-to-head
(`docs/stage2_architecture_variants_draft.md` strand 14): combining them
made H=720's MAE strictly worse (+3.1%) for no extra MSE gain over
Huber alone — the two compete for the same job (MSE's outlier
sensitivity) rather than compounding. λ=-2 is a far more extreme version
of Yeo-Johnson than the MLE-fit lambda tested against Huber; there's no
reason to expect the conflict gets milder at a more extreme setting, and
good reason to expect it gets worse. **Don't combine Huber with any
Yeo-Johnson variant.**

### A technical gotcha for the `DLinear` ensemble specifically

Averaging predictions only makes sense if both models are in the same
numeric space. `SegRNN` and `DLinear` are currently both trained on
plain `StandardScaler`d data. If a future combo puts Yeo-Johnson on the
`SegRNN` side but not `DLinear`'s, their raw predictions live on
different scales — averaging them directly wouldn't just be
suboptimal, it would be numerically meaningless. Either inverse-transform
both back to a common scale first, or keep preprocessing identical across
both models being ensembled.

### Combinations worth building, grouped by what they're chasing

**A — the MAE chase (your example).** `SegRNNPoolContext` + Yeo-Johnson
λ=-2 (`--model SegRNNPoolContext --power_transform 4 --yj_lambda -2`).
Both push MAE the same direction, at different pipeline levels
(preprocessing vs. architecture), with no known mechanism conflict —
unlike Huber, pool context doesn't touch the loss function or the
outlier-sensitivity mechanism λ=-2 is manipulating.

Expectation, stated plainly before running it: λ=-2 alone already gets
MAE to 0.2832 by doing something fairly extreme to the loss landscape
(MSE nearly doubles as the cost). Pool context's own MAE effect alone is
much smaller (-0.8% to -1.0%, vs. λ=-2's -34.5%). Even if the two
compound perfectly — not guaranteed — pool context's contribution is a
rounding error next to what λ=-2 is already doing: roughly
0.2832 × 0.99 ≈ 0.280, not 0.2. Reaching 0.2 needs something that
changes the mechanism, not a small correction stacked on top of it.
Worth running to see whether it's additive at all, but the "wow result"
framing should be set before the run, not chased after it.

**B — the balanced win (the stronger headline candidate for the
report).** Huber loss + `SegRNNPoolContext` (`--model SegRNNPoolContext
--loss huber`). Both are this project's only two no-trade-off wins
(Huber: -4.4%/-3.3% MSE/MAE at H=720; pool context: -0.7% to -1.8% MSE
*and* -0.8% to -1.0% MAE at 3/4 horizons), touch different parts of the
pipeline, and have no known conflict with each other. This is the
strongest candidate for "our best combined model" in the actual report —
safer than the λ=-2 chase, and the one combination plausible enough to
beat the reconstruction on *both* metrics at *most* horizons at once,
something no single strand has done across all four horizons.

**C — efficient and accurate.** `d_model=256` + `SegRNNPoolContext` +
Huber loss, all three stacked (fully free, no new code). If B holds up,
this checks whether the same combined win survives at 25.5% of the
parameters — the practical "ship this" configuration if it does.

**D — lower priority, not core to the report.**
- Window stats in any combination: the H=720 catastrophe (+9.2% MSE,
  +8.4% MAE) makes it risky in anything meant to hold up across all four
  horizons — if used, scope the claim to H<=336 explicitly.
- Encoder direction: "roughly neutral" is the only characterization on
  record, no precise numbers transcribed yet — not enough signal to
  build a targeted combination around.
- `DLinear` ensemble + Huber (`SegRNN` trained with `--loss huber`, then
  averaged post-hoc with `DLinear`): free to combine, no known conflict,
  but the ensembling effect alone was already small — likely a marginal
  addition on top of B, not a headline on its own.

### Recommended build order

1. **B** (Huber + pool context) — highest expected value, safest bet,
   directly reportable as "our best model."
2. **C** (add `d_model=256` on top of B) — if B works, immediately worth
   checking the efficient version.
3. **A** (the λ=-2 MAE chase) — worth doing for the report's honesty
   (the question was asked, the answer belongs in the writeup either
   way), but go in with the tempered expectation above, not the 0.2
   target.
4. Skip window-stats and encoder-direction combinations unless B/C/A
   leave time — the evidence for including either is weak enough that
   they're more likely to dilute a combination than strengthen it.
