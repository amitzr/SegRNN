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

## 2. Add stronger time-series features

| Idea | Performance | In lectures? | Recommend for final combo? |
|---|---|---|---|
| Calendar features in positional embedding (3 attempts: encoder x2, decoder) | Regressed at every horizon, every attempt, seed-confirmed real | Partial (`DL_for_TS.pdf` embedding-layer slide covers embedding categorical/cyclical features generally, not this specific use) | **No** |
| ROCKET random-convolution features | Regressed at 3/4 horizons; H=336's +8.7% MSE is the worst single-horizon regression in the project | Yes, as a technique (`Pre-precessing.pdf`), but classification-era, adapted to regression without precedent | **No** |
| Top-K FFT-magnitude features (with and without Yeo-Johnson) | Qualitatively "terrible" (exact table not yet transcribed) | Partial (`Pre-precessing.pdf` covers window/frequency features generally, not top-K FFT specifically) | **No** |

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
