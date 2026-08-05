# SegRNN project — results summary

Everything tried, with numbers. Written as a standalone handoff: it
assumes no prior context.

**Paper:** Lin et al., "SegRNN: Segment Recurrent Neural Network for
Long-Term Time-Series Forecasting", IEEE IoT-J vol. 13 no. 5, 2026.
Fork of the authors' official code (`lss-1138/SegRNN`, Apache-2.0).

**Stage 2 test bed.** Unless stated otherwise, every Stage 2 number below
is **ETTh1, multivariate, look-back L=720**, against this baseline
(the project's own reconstruction, which matches the paper to 0.01–0.14%):

| Metric | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| MSE | 0.3510 | 0.3925 | 0.4233 | 0.4657 |
| MAE | 0.3925 | 0.4142 | 0.4327 | 0.4719 |

**Protocol.** 6:2:2 chronological split, `StandardScaler` fit on train
only, 30 epochs, Adam, LR decay 0.8 after epoch 3, early stopping
patience 5. Every strand tested independently against the baseline —
never stacked with another untested strand.

**Caveat that applies to every number here:** all runs are **single-seed
(2024)**. The only seed-variance data collected is in the multi-seed rows
below, which put run-to-run std at ~0.0009 (H=336) and ~0.0050 (H=720)
in MSE. Differences smaller than roughly ±1% should be treated as
unresolved, not as measured effects.

---

## Stage 1 — reconstruction (the baseline everything is measured against)

Reproducing the paper's Table II (multivariate) and Table III
(univariate) for ETTh1, ETTh2, ETTm1, ETTm2 — 32 cells.

| Table | Cells within 2% | Mean abs Δ MSE | Mean abs Δ MAE | Mean signed Δ MSE |
|---|---|---|---|---|
| II (multivariate) | **16 / 16** | 0.33% | 0.23% | +0.11% |
| III (univariate) | **15 / 16** | 0.55% | 0.42% | +0.31% |

Mean signed error near zero on both = imprecision, not systematic bias.
The one failure is **ETTm2 univariate H=720 (+5.64% MSE, +4.11% MAE)**,
which remains unexplained: it is not the data file (the same `ETTm2.csv`
reproduces multivariate H=720 to −0.53%), and enabling RevIN made it
worse, not better (+8.10%).

### Stage 1 secondary findings

| Finding | Evidence |
|---|---|
| **The paper's stated config does not reproduce the paper's own tables.** Section V-A2 calls `seg_len=48` and `d_model=512` "uniform"; the released scripts use `seg_len=24` for ETTh1/ETTh2 and `d_model` 256/128 in several univariate runs | Over the 21 cells where the two disagree: script config **0.22%** mean abs Δ MSE vs. paper-stated config **3.90%** — an 18× gap. The tables are reproducible from the repository, not from the paper |
| **Hyperparameter sensitivity grows sharply with horizon** | Paper-config mean abs Δ MSE by horizon: 1.49% / 1.99% / 2.69% / **8.50%** at H=96/192/336/720 |
| **RevIN is worth ~14–18% MSE at one specific cell** | ETTm1 univariate H=720: turning it on took the cell from +13.94% to −0.01%; turning it off cost +18.02%. Two independent runs, opposite directions |
| **The paper's own architecture description beats its released script on ETTm2** | `channel_id=1` (CP encoding enabled, as the architecture section describes but the script does not) beats the *published* ETTm2 multivariate numbers at H=336 (−1.53%) and H=720 (−5.66%), while losing at H=96/192. **Untested lead, single seed** |

---

## Table 1 — Going into the final submission

Seven strands. Two of them are negative or mixed results that are
included **because** they are informative, not because they won.

| # | Strand | Category | Result | Verdict |
|---|---|---|---|---|
| 1 | **Yeo-Johnson power transform** (MLE-fit λ) | Preprocessing | MSE **−7.4 / −7.3 / −4.6 / −3.6%**; MAE **+4.1 / +5.5 / +6.7 / +4.4%** | **Best MSE in the project**, at a real MAE cost. A trade-off, not a clean win |
| 2 | **Yeo-Johnson fixed-λ sweep** (H=336/720) | Preprocessing | λ=−2: MSE **+85 / +89%**, MAE **−34.5 / −25.5%**. No fixed λ beats the MLE fit on MSE | **Best MAE in the project** (0.2832 at H=336) at the worst MSE in the project. λ=1 exactly reproduces the baseline, confirming the implementation |
| 3 | **Multi-scale window statistics** | Time-series features | MSE −0.0 / **−1.7** / **−2.8** / **+9.2%**; MAE −0.0 / −0.0 / +0.3 / **+8.4%** | **Mixed.** Real win at H=192/336; the H=720 regression is the worst single number in the project. Scope any claim to H≤336 |
| 4 | **AIC/BIC → `d_model`** | Efficiency | `d_model=256`: MSE +1.3 / +0.8 / **+0.35** / +4.8%, at **25.5% of the parameters** (408k vs 1.60M) | **Clearest efficiency result.** AIC/BIC independently reject `d_model=512` at every horizon and favour 64/128 — a free post-hoc analysis, not a trained model |
| 5 | **Huber loss** | Training objective | H=336: MSE +0.9%, MAE −0.5%. H=720: MSE **−4.4%**, MAE **−3.3%** | **Cleanest win in the project.** Both metrics improve at H=720 with no trade-off |
| 6 | **Parameter-free pooling context** (mean-pool) | Architecture | MSE **−0.7 / −1.8 / 0.0 / −1.6%**; MAE **−0.9 / −0.8 / +1.0 / −1.0%** | **Cheapest win.** Both metrics improve at 3 of 4 horizons at **zero added parameters**. ⚠️ Tested with **mean** pooling — max pooling was never run |
| 7 | **Encoder direction** (bidirectional GRU) | Architecture | Qualitatively "around the same" — **exact per-horizon figures were never recorded** | Neutral at real added parameter cost. Included as a negative result; the surprise is that the heaviest strand by parameter count did not clearly regress |
| 8 | **Post-hoc `SegRNN`+`DLinear` ensemble** | Model tuning | MSE +0.6 / +0.1 / −0.1 / **−2.2%**; MAE **−1.0 / −0.6 / −0.4** / +0.8% | Small and mixed, but it **reversed the sign** of the failed jointly-trained linear shortcut (Table 2 #11) by training both models independently and averaging predictions, not metrics |

(Eight rows for seven chosen strands — Yeo-Johnson is split into its
MLE-fit and fixed-λ-sweep halves, which were separate experiments.)

### Known conflict — do not combine

**Huber loss + Yeo-Johnson.** They compete for the same job (MSE's
outlier sensitivity) rather than compounding:

| Config | H=336 MSE / MAE | H=720 MSE / MAE |
|---|---|---|
| Baseline | 0.4233 / 0.4327 | 0.4657 / 0.4719 |
| Huber alone | 0.4271 (+0.9%) / 0.4306 (−0.5%) | **0.4453 (−4.4%) / 0.4563 (−3.3%)** |
| Huber + YJ | 0.3983 (−5.9%) / 0.4526 (+4.6%) | 0.4448 (−4.5%) / **0.4865 (+3.1%)** |

At H=720 the combination gains essentially nothing on MSE over Huber
alone and flips MAE from −3.3% to +3.1%. Strictly worse.

---

## Table 2 — Tried, did not work

| # | Strand | Category | Result | Why it failed / notes |
|---|---|---|---|---|
| 1 | **Calendar features** (3 attempts: encoder ×2, decoder) | Features | MSE +0.6 / +2.0 / +2.1 / +3.9% (best variant: hour embed, encoder); worst variant +3.4 / +1.3 / +0.2 / +2.1% | Regressed at every horizon on every attempt. **Seed-confirmed real** (H=336: +0.0042 ± 0.0030; H=720: +0.0074 ± 0.0027 over n=3) — the only failure verified against seed noise |
| 2 | **RevIN**, `affine=False` | Preprocessing | MSE **+4.6 / +3.6 / +3.4 / +3.1%**; MAE +2.0 / +2.0 / +1.6 / +1.2% | Worse at all four horizons. The paper uses RevIN only for datasets with severe distribution shift, not ETTh1 |
| 3 | **RevIN**, `affine=True` | Preprocessing | Better than `affine=False`, still net worse than no RevIN — exact figures not recorded | Follow-up to #2; did not rescue it |
| 4 | **Attention over encoder states** | Architecture | MSE **+1.5 / +0.5 / +5.1 / +3.5%**; MAE +1.3 / +2.1 / +3.9 / +3.0% | Worse at all four horizons. Later shown to be the **added Q/K/V parameters**, not the bottleneck-removal idea — parameter-free pooling (Table 1 #6) does the same job and wins |
| 5 | **ROCKET random-convolution features** | Features | MSE **+5.2 / +1.3 / +8.7 / −0.15%**; MAE +2.6 / +1.5 / +6.6 / +1.4% | H=336's +8.7% was the largest single-horizon regression until window stats at H=720. Classification-era technique with no forecasting precedent |
| 6 | **Top-K FFT-magnitude features** | Features | Qualitatively "terrible", worst in its batch — **exact figures never recorded** | Same `h_n`-injection pattern as #5 |
| 7 | **Encoder depth** (stacked GRU, 2 layers) | Architecture | Qualitatively "slightly worse" — **exact figures never recorded** | Adds capacity; the project's consistent finding is that adding capacity hurts |
| 8 | **`Conv1d` value embedding** | Architecture | Qualitatively "bad" — **exact figures never recorded** | Matches ISMRNN (arXiv 2407.10768), which reports conv hurting on all datasets except Weather |
| 9 | **Un-shared decode cell** | Architecture | Regresses, and the gap **grows with horizon** — exact figures never recorded | Combined with the neutral bidirectional result, suggests bidirectional context was compensating for the un-shared decoder's cost, not that un-sharing is free |
| 10 | **Weight-tied embed/predict** | Architecture | MSE consistently slightly worse; MAE roughly tied, small win at H=720 — exact figures never recorded | Weakest support for the "reduce capacity" hypothesis that `d_model=256` and Huber both confirmed |
| 11 | **Linear shortcut**, jointly-trained blend gate | Architecture | Worse — exact figures never recorded | Superseded by the post-hoc DLinear ensemble (#17), which changed the sign of the result |
| 12 | **LayerNorm after input embedding** | Architecture | Worse — exact figures never recorded | Superseded by #13 |
| 13 | **LayerNorm on `h_n`** (before decode) | Architecture | MSE +2.9 / +0.3 / +2.8 / **−4.7%**; MAE +1.7 / +1.0 / +1.2 / **−2.7%** | **Mixed, not a failure.** Clean win on both metrics at H=720 only. Worth reconsidering for a long-horizon-only claim |
| 14–16 | **Frozen recurrent cell / Echo State Network** (naive frozen; ESN spectral-radius 0.9; radius sweep 0.5/0.9/0.99/1.1) | Architecture | All three "did not improve" — **exact figures never recorded** | Reservoir computing appears nowhere in the paper or the lectures. The most aggressive version of the "reduce trained capacity" lever; it crossed the line the milder reductions did not |
| 17 | **Multi-seed prediction ensembling** | Model tuning | H=336: MSE −0.26%, MAE −0.16%. H=720: MSE +0.04%, MAE +0.13% | Borderline noise at H=336, clearly noise at H=720. Its real value was measuring **seed std: 0.0009 (H=336), 0.0050 (H=720)** |
| 18 | **Blended MSE+MAE loss** | Training objective | H=336: MSE +0.6%, MAE −0.5%. H=720: MSE −1.3%, MAE −1.2% | Works, but is a strictly weaker version of Huber loss. No reason to report both |
| 19 | **Blended loss + Yeo-Johnson** | Combination | H=336: 0.3983 (−5.9%) / 0.4530 (+4.7%). H=720: 0.4508 (−3.2%) / 0.4884 (+3.5%) | Same competing-fixes pattern as Huber+YJ |
| 20–21 | **Box-Cox** (MLE-fit and fixed-λ sweep) | Preprocessing | **Dropped — never produced a usable number** | Built on the assumption that ETTh1 is strictly positive. It is not. sklearn rejected the MLE run outright; the custom fixed-λ scaler had no positivity check, silently produced `NaN`, and wasted a full 30-epoch run. Yeo-Johnson handles negatives natively, which is why it is the project's preprocessing result instead |

---

## Data-quality warning for whoever writes this up

**Eleven strands have no exact per-horizon figures recorded** — only
qualitative verdicts ("worse", "terrible", "around the same"): Table 2
rows 3, 6, 7, 8, 9, 10, 11, 12, 14–16, and Table 1 row 7 (encoder
direction). They were run and the outcome observed, but the numbers were
never transcribed.

Any of these that need to appear in the report with numbers must be
re-run. The two most likely to matter:

- **Encoder direction** (Table 1 #7) — it is in the submission set and
  currently has no numbers at all.
- **The three reservoir/ESN strands** — the *shape* of the failure
  (does ESN tuning beat naive freezing? is there radius sensitivity?) is
  more informative than the headline regression, and none of it was
  captured.

---

## Combinations proposed but never built

None of these were run.

| Combination | Rationale | Cost |
|---|---|---|
| **A** — pooling context + Yeo-Johnson λ=−2 | Both push MAE the same direction at different pipeline levels. Expectation set in advance: ≈0.280 MAE, **not** 0.2 — pooling's −1% is a rounding error next to λ=−2's −34.5% | Flags only |
| **B** — Huber + pooling context | The project's only two no-trade-off wins, touching different parts of the pipeline, no known conflict. **Strongest candidate for "our best model"** | Flags only |
| **C** — B + `d_model=256` | Checks whether B survives at 25.5% of the parameters | Flags only |
| **D** — Huber + pooling + LayerNorm-`h_n` | The three strands that clean-win at H=720. Mechanisms are distinct (loss / bottleneck / hidden-state scaling) | **Needs a new hybrid model class** — pooling and LayerNorm live in separate files and only one `--model` can be selected per run |

**Build order if resumed:** B → C → A. Skip window stats and encoder
direction in combinations; the evidence for either is too weak to
strengthen a combination.

---

## Open methodological gaps

1. **Single seed throughout.** `CLAUDE.md` specifies three seeds
   (2021/2022/2023) reported as mean ± std. Everything ran at the
   upstream default, seed 2024. Only calendar features and multi-seed
   ensembling have any seed-variance data.
2. **Determinism is not pinned.** Upstream sets `random.seed`,
   `torch.manual_seed`, `np.random.seed` — but never
   `torch.backends.cudnn.deterministic`, anywhere in the repository. The
   model is a GRU, and cuDNN's RNN backward is nondeterministic by
   default. Run-to-run variation was empirically below observable
   precision, but it is not guaranteed by the code.
3. **One unreproduced Stage 1 cell** (ETTm2 univariate H=720), cause
   unknown.
4. **Evaluation metrics.** Only MSE/MAE were used. Lecture-taught
   alternatives worth adding: **RMSE** and **MdAE** (cheap, and MdAE ties
   into this project's recurring MSE-vs-MAE story). MAPE/SMAPE/RMSLE/NMSE
   are poor fits for standardized data containing non-positive values.
