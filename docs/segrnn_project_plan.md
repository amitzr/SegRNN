# SegRNN Final Project — Execution Plan (v2, verified against the paper)

**Paper:** Lin, Lin, Wu, Zhao, Mo, Zhang. "SegRNN: Segment Recurrent Neural Network for Long-Term Time-Series Forecasting." *IEEE Internet of Things Journal*, vol. 13, no. 5, pp. 9861–9871, March 2026. DOI 10.1109/JIOT.2025.3647705. (Earlier preprint: arXiv:2308.11200.)
**Official repo:** github.com/lss-1138/SegRNN — Apache-2.0.
**Task type:** multivariate, supervised, long-term forecasting (LTSF). Channel-Independent (CI) strategy.
**Course topics covered by the paper itself:** fully connected layers, RNN/GRU/LSTM, encoder–decoder, positional embeddings (attention lineage). Only the CNN bullet is absent — which our improvement plan optionally closes.

---

## 0. What the paper actually contains (verified — do not re-propose these as "improvements")

### Architecture (Section IV + Algorithm 1)
- **Encoding:** segment the lookback `X ∈ R^{L×C}` into `n = L/w` non-overlapping segments of length `w`; project each with `Linear(w → d)` + ReLU; feed the `n` embeddings through a **single-layer GRU**; keep the final hidden state `h_n`.
- **Decoding (PMF):** repeat `h_n` `m = H/w` times, pair each copy with a positional embedding `pe`, run them **in parallel through the same GRU cell**, then `Dropout` → `Linear(d → w)` → reshape to `H`.
- **Positional embedding:** `pe = concat(rp, cp)`, each of dim `d/2`, both **learnable**. `rp` = relative position of the future segment; `cp` = channel identity. **No timestamps anywhere** — Algorithm 1's only data input is `X`.
- **Normalization:** subtract the last value of the input before encoding, add it back after decoding; **RevIN** additionally for datasets with severe distribution shift (Traffic).
- **Loss:** L2 / MSE (Eq. 7).

### Configuration (Section V-A2)
Look-back **L = 720**, segment length **w = 48** (chosen as the largest common divisor of 720 and all horizons → n = 15 iterations), **1 GRU layer**, hidden size **d = 512**, Adam, **30 epochs**, LR decay ×0.8 after the first 3 epochs, early stopping patience 5. Dropout/batch size/LR vary per dataset (see repo scripts).

### Data & protocol (Table I)
| Dataset | Channels | Frequency | Timesteps | Split |
|---|---|---|---|---|
| ETTh1 & ETTh2 | 7 | 1 hour | 17,420 | 6:2:2 |
| ETTm1 & ETTm2 | 7 | 15 min | 69,680 | 6:2:2 |
| Electricity | 321 | 1 hour | 26,304 | 7:1:2 |
| Solar-Energy | 137 | 10 min | 52,560 | 7:1:2 |
| Traffic | 862 | 1 hour | 17,544 | 7:1:2 |
| Weather | 21 | 10 min | 52,696 | 7:1:2 |

Splits are chronological. Horizons H ∈ {96, 192, 336, 720}. Metrics: **MSE and MAE only**.
Baselines used: iTransformer, PatchTST, DLinear, FEDformer, Autoformer, Informer, DeepAR, GRU.
Hardware in the paper: two NVIDIA T4 GPUs, 16 GB — i.e. **Colab-equivalent**. This is the single best feasibility signal.

### Ablations the authors already did → these belong in **Stage 1 (reproduction)**, not Stage 2
| Paper artifact | What it shows |
|---|---|
| **Fig. 6** — segment length sweep on ETTm1 (L=H=192), incl. **w = 1** | w=1 recovers the point-wise RNN (worst); error falls as w grows; w→L degenerates to an MLP and error rises again; inference time falls monotonically with w |
| **Fig. 7** — PMF vs RMF | PMF better and more stable, advantage grows with horizon; PMF slower below H=192, faster above |
| **Table IV** — non-overlapping vs overlapping segmentation | Overlapping is consistently **worse** (redundant info + deeper recurrence). Also reports extreme-value counts: Traffic ≈ 23.8 per channel (|z| > 6), Electricity 1.4, Solar-Energy 0 |
| **Table V** — PE ablation (RP+CP / RP / CP / None) | **RP alone cuts MSE 28.8% vs no PE** — the PE is the highest-leverage component. CP helps generally but *hurts* on Traffic (>800 channels) |
| **Fig. 8** — RNN vs LSTM vs GRU cell | GRU best. Repo ships `scripts/SegRNN/ablation/rnn_variants.sh` |
| **Table VI** — SegRNN vs PatchTST efficiency | >78% less training time, >82% less max GPU memory; SegRNN d=512 vs PatchTST d=128 |

### What is genuinely absent (→ legitimate Stage 2 territory)
1. **Timestamps in any form** (no hour-of-day / day-of-week / month).
2. **Any probabilistic output** — point forecasts only, MSE loss only.
3. **Naive / seasonal-naive baselines**, and any scale-free metric (no MASE/sMAPE).
4. **Multiple seeds / standard deviations** — main tables are single runs.
5. **A hidden-size sweep** — d = 512 is fixed and never justified, despite the paper's IoT/resource-constrained framing.
6. **Directionality of the RNN** — Fig. 8 varies the *cell type*, never bidirectionality.

---

## 1. Novelty audit — has anyone already done our improvements?

Checked the literature citing SegRNN. Summary: **no**, but name these relatives in the report's related-work paragraph. Being second on a technique is fine for a course project; pretending to be first is not.

| Our improvement | Closest published work | What they did | Overlap |
|---|---|---|---|
| Calendar embeddings in the PE | **CycleNet** (NeurIPS'24 Spotlight, *same lab*) | Residual Cycle Forecasting: learnable recurrent cycles model periodicity, backbone predicts the residual; plug-and-play, shown on PatchTST/iTransformer | Same goal (inject periodicity), different mechanism & level; never applied to SegRNN |
| " | **GLAFF** (NeurIPS'24) | Timestamps modeled separately (attention mapper + robust denormalizer + adaptive combiner) as a model-agnostic plugin | External wrapper, not an in-model PE change |
| " | **D2Vformer** (arXiv 2409.11024) | Date2Vec time-position embedding using both input and *future* date matrices, explicitly leakage-free | Same leakage argument, Transformer backbone |
| " | **IndexNet** (arXiv 2509.23813) | Learnable embedding sets per temporal field (24 hour-of-day vectors, 7 day-of-week, etc.), zero-init | Mechanically closest; standalone model |
| Conv1d segment embedding | **ISMRNN / MSegRNN** (arXiv 2407.10768) | Attacks the *same component*: replaces explicit segmentation with "implicit segmentation" (dual linear projections) + residual encoder + Mamba preprocessing | Same target, different tool. **Their Appendix F found the conv layer hurt on all datasets except Weather** → treat conv as higher-risk |
| Bidirectional GRU encoder | — | Paper's Fig. 8 varies cell type only | **No overlap found** |
| Quantile / probabilistic head | — | No probabilistic SegRNN variant found | **No overlap found** |
| Hidden-size sweep | ISMRNN reports time/memory vs SegRNN | Efficiency comparison, not a d-sweep | Minimal |

Also worth citing for context: **P-sLSTM** (AAAI'25 — patching + channel independence on sLSTM), **TQNet** (ICML'25, same lab — periodically shifted learnable attention queries; **claims the inter-channel/CP future work, so do not propose "fix CP encoding"**), **WITRAN** (RNN-based LTSF with adaptive cycles).

---

## 2. Stage 0 — repo verification (~1–2 hours, do before committing)

1. `git clone https://github.com/lss-1138/SegRNN`. README specifies `conda create -n SegRNN python=3.8` + `pip install -r requirements.txt`. On Colab skip conda; pip-install only what's missing.
2. **Datasets:** the Autoformer Google Drive bundle linked in the README. Place CSVs directly in `./dataset/` (e.g. `./dataset/ETTh1.csv`). Store on your Google Drive and mount it so you download once.
3. Open `scripts/SegRNN/etth1.sh` and copy the exact flags — these *are* the paper's hyperparameters. Record `seq_len`, `pred_len`, `seg_len`, `d_model`, `dropout`, `learning_rate`, `batch_size`, `train_epochs`, RevIN flag.
4. **Smoke test:** run ETTh1 / pred_len 96 for 1–2 epochs. Confirm it trains, evaluates, prints MSE/MAE. Time one epoch — this calibrates the whole budget.
5. Read `models/SegRNN.py` line by line and map every line to Algorithm 1 in the paper. You must be able to explain all of it in Section 1 of the report.
6. Check `data_provider/` — the Autoformer-derived loader normally returns time-feature tensors (`batch_x_mark`, `batch_y_mark`). **Confirm whether they exist and whether SegRNN ignores them.** If they exist, Improvement A is mostly wiring. If not, you build them with `pandas.DatetimeIndex`.
7. **Note the known bug fix:** the README documents a long-standing `drop_last` bug (last test batch was dropped) that was fixed in `data_provider/data_factory.py` and `exp/exp_main.py`. Verify you have the fixed version and say so in the report — this is a preprocessing/data-quality point.
8. Run one full config to completion and compare against Table II. Landing within ~5–10% de-risks the project.

**Careful about which setting you reproduce:** the paper's tables use **L = 720**. The repo also has `scripts/SegRNN/Lookback_96` and a separate README table for L = 96. Reproduce **L = 720** (the paper), and mention L = 96 only as context.

---

## 3. Scope (Colab-aware)

- **Core:** ETTh1, ETTm1, Weather — all 4 horizons, **3 seeds** (2021/2022/2023), mean ± std.
- **Stretch:** Electricity (321 channels) if budget allows; it strengthens the multivariate story and is where calendar features should help most.
- **Skip:** Traffic (862 channels) and Solar-Energy unless everything else is finished.
- State explicitly in the report: task is multivariate supervised forecasting; per-dataset sampling frequency (Table I above); input window 720; output windows 96/192/336/720; segment length 48; all hyperparameters from the official scripts.
- Fix seeds for `torch`, `numpy`, `random`, and set cudnn deterministic. Log every run to a CSV: `run_id, model, dataset, horizon, seed, flags, MSE, MAE, MASE, epoch_time, params, peak_mem`.

---

## 4. Stage 1 — Reconstruction

1. **Pipeline** (document, don't rewrite unnecessarily): loading, datetime parsing, chronological split (6:2:2 ETT / 7:1:2 others), standardization **fitted on train only**, sliding windows, last-value normalization vs RevIN.
2. **Baselines the paper lacks (required by the assignment):**
   - Naive: repeat the last observed value across the horizon.
   - Seasonal naive: repeat the last full daily cycle (period 24 for ETTh/Electricity, 96 for ETTm, 144 for Weather).
   - Optional: single linear layer (DLinear-style), ~20 lines.
   Same windows, same splits, same scaling.
3. **Metrics:** MSE + MAE (paper's, explain what each measures and that they're computed on standardized data), plus **MASE** (scaled by in-sample seasonal-naive error; explain why it's scale-free and comparable across datasets).
4. **Reproduce the authors' own ablations** — this is where reproduction becomes impressive:
   - Segment-length sweep including **w = 1** (their Fig. 6). Reproducing the point-wise-RNN failure with your own numbers is the strongest possible demonstration that you understand the paper's thesis.
   - PMF vs RMF (Fig. 7).
   - RNN variants via `scripts/SegRNN/ablation/rnn_variants.sh` (Fig. 8).
5. **Comparison table:** paper's numbers vs yours (mean ± std) per dataset × horizon, baselines included. Then a differences paragraph: seeds, hardware, library versions, reduced dataset subset, epochs.

---

## 5. Stage 2 — Improvements (three headlines, one per success criterion)

### A. FEATURES (accuracy bet) — calendar embeddings in the positional embedding
**What:** extend `pe = concat(rp, cp)` to `pe = concat(rp, cp, tp)` where `tp` is built from learnable `Embedding` tables indexed by the actual timestamp of that future segment — hour-of-day (24), day-of-week (7), and month (12) where the frequency warrants it. Optionally also add a timestamp embedding to each *input* segment.
**Why (course-grounded):** this is the lecture's improved airline-passenger model — "include the year and the month as predictor variables" → `Embedding(12, 5)(cat_layer)` concatenated with other inputs. **Why (paper-grounded):** Table V shows the PE is the highest-leverage component (RP alone = 28.8% MSE reduction), and `rp`/`cp` are purely index-based. The decoder currently knows *"you are future segment #3"* but not *"you are 08:00 Monday."*
**Leakage:** none — future timestamps are known at prediction time. State this explicitly (D2Vformer makes the same argument).
**Hypothesis to state up front:** largest gains on ETTm1 and Electricity (strong daily/weekly cycles), smallest on Weather. Check it.
**Honesty note for the report:** the technique is established (CycleNet, GLAFF, D2Vformer, IndexNet); the contribution is testing it inside SegRNN's PE, which is untested.

### B. ARCHITECTURE (safe) — bidirectional GRU encoder
**What:** make the encoder GRU bidirectional; concatenate the forward and backward final states (2d) and project back to d with a linear layer before the PMF decoder (the decoder must keep dim d since it shares the cell).
**Why (course-grounded):** the `bidirectional` flag is on your RNN-in-PyTorch slide and the lecture's Keras example uses `Bidirectional(LSTM(20))`. **Why (paper-grounded):** Fig. 8 varies the cell type but never directionality — a real gap. **No leakage:** the entire lookback window is past data; reading it in reverse uses no future information. Say this explicitly, since it looks suspicious at first glance.
**Secondary (higher-variance) variant:** replace `Linear(w→d)+ReLU` with a `Conv1d` over each segment (optionally dilated/causal), closing the CNN lecture bullet. Flag the risk: ISMRNN attacked this same component with linear implicit segmentation, and their Appendix F found conv hurt on all datasets but Weather. Run it as an ablation with a stated hypothesis either way.

### C. EFFICIENCY (safe) — is d = 512 necessary?
**What:** sweep hidden size d ∈ {512, 256, 128, 64}, reporting MSE/MAE **alongside** parameter count, training time per epoch, and peak GPU memory — extending the paper's own Table VI methodology.
**Why:** the paper's entire framing is resource-constrained IoT, yet it fixes d = 512 while its main rival PatchTST uses 128 (Table VI caption). If accuracy holds at d = 128, you've made SegRNN materially lighter with a one-flag change — a clean, quantified efficiency result.

### D. OPTIONAL STRETCH — uncertainty
Replace `Linear(d → w)` with three quantile heads (P10/P50/P90) trained with **pinball loss** instead of MSE; report empirical coverage vs nominal 80%. Use P50 as the point forecast for MSE/MAE comparability. Not covered by the lecture (which only teaches MSE/MAE/RMSE), so budget learning time — but no published probabilistic SegRNN exists, and DeepAR (their own baseline) does this, which makes the gap easy to motivate.

### Coverage of the instructor's improvement categories
| Category | Covered by |
|---|---|
| Improve preprocessing and data quality | Train-only scaling; `drop_last` bug fix; last-value-norm vs RevIN standardized across datasets (partial/supporting) |
| Add stronger time-series features | **A — calendar embeddings** |
| Improve or tune the forecasting model | Hidden-size + segment-length sweeps; 3-seed protocol |
| Improve ML or deep-learning architecture | **B — bidirectional GRU (+ optional Conv1d segment embedding)** |
| Improve anomaly or change-point detection | Not applicable (forecasting paper). Adjacent only: the paper's extreme-value analysis (Table IV) motivates the RevIN discussion — robustness, not detection. Do not overclaim. |
| Improve probabilistic forecasting and uncertainty | **D — quantile head (optional)** |
| Improve computational efficiency | **C — hidden-size sweep with params/time/memory reported** |

---

## 6. Week-by-week (≈6 weeks)

- **Week 1:** Stage 0 checklist; read the paper twice; annotate `models/SegRNN.py` against Algorithm 1; one full ETTh1 run matched to Table II; set up Drive caching + run-log CSV.
- **Week 2:** full reconstruction on core datasets × 4 horizons × 3 seeds; implement naive/seasonal-naive/linear baselines + MASE. **Freeze Stage 1 numbers.**
- **Week 3:** reproduce the paper's ablations (w-sweep incl. w=1, PMF vs RMF, RNN variants). Start Improvement A (calendar embeddings); debug on ETTh1/96.
- **Week 4:** full runs for A and B across core datasets/horizons, 3 seeds each.
- **Week 5:** Improvement C sweep; optional D or the Electricity stretch; assemble tables + figures (prediction-vs-actual plots as in the lecture; error-vs-horizon curves; accuracy-vs-cost scatter for C).
- **Week 6:** PDF report, README (env, package versions, exact commands, expected outputs), code cleanup, dataset link + instructions.

---

## 7. Report mapping (assignment's required structure)

1. **Original architecture** — segment partition + `Linear(w,d)`+ReLU, single-layer GRU encoder, PMF decoder with RP/CP positional embeddings, last-value normalization (+RevIN), MSE loss, hyperparameters (L=720, w=48, d=512, 30 epochs, Adam, LR decay 0.8, patience 5).
2. **Paper results** — Table II/III excerpts for your datasets; note SegRNN ranks top-2 in 54/64 multivariate metrics with 34 firsts, and improves 75%/78% in MSE over GRU/DeepAR.
3. **Reconstruction results** — your table (mean ± std), baselines, MASE, plus your reproductions of Figs. 6–8; differences paragraph.
4. **Improved architecture** — A, B, C (+D), each with course grounding, paper grounding, leakage argument, and a stated hypothesis. Include the related-work paragraph naming CycleNet / GLAFF / D2Vformer / IndexNet / ISMRNN and what each did.
5. **Improved results** — three-way table (paper / reconstruction / improved) on identical splits, seeds, and metrics; plus the efficiency table for C.
6. **Discussion** — which hypotheses held; seed variance vs claimed improvements; what the naive baselines revealed; whether conv helped (and how that compares with ISMRNN's Appendix F); limitations.
7. **References** — SegRNN (IEEE IoT-J 2026 version), DLinear, PatchTST, iTransformer, RevIN, CycleNet, GLAFF, D2Vformer, IndexNet, ISMRNN, DeepAR, dataset sources.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Python 3.8 / package friction on Colab | Install from `requirements.txt` selectively; `models/SegRNN.py` is small enough to port into a clean notebook as a fallback |
| Numbers don't match the paper | Check `seq_len=720` (the paper's edge depends on the long lookback), the RevIN flag, the `drop_last` fix, and that metrics are on standardized data. A 5–10% gap with explanation is acceptable per the assignment |
| Calendar embeddings show no gain | Expected on Weather; the ETTm/Electricity contrast **is** the finding. B and C cannot fail, so Stage 2 is never empty |
| Conv embedding hurts | Predicted by ISMRNN's Appendix F — state the hypothesis in advance so a negative result is a result |
| Colab disconnects | Checkpoint per epoch to Drive; keep runs short (core datasets are minutes-scale); never rely on session state |
| Scope creep | One improvement per success criterion (accuracy / architecture / cost). Quantile head only if weeks 1–4 land on time |
