# Project context

University final project: reconstruct the SegRNN paper, then improve it.
Paper: Lin et al., "SegRNN: Segment Recurrent Neural Network for Long-Term Time-Series Forecasting", IEEE IoT-J vol. 13 no. 5, pp. 9861-9871, 2026.
This repo is the authors' official code. Compute: Google Colab (T4). The paper itself used two T4s, so the budget is realistic.

## Ground rules
- Never modify results or fabricate numbers. Every number in the report must come from a run I can re-execute.
- Respect temporal validity: no future information in training, preprocessing, scaling, or tuning. Scaling is fit on the train split only.
- Reproduce the paper's setting: lookback L=720 (NOT the L=96 scripts), seg_len=48, 1 GRU layer, d_model=512, 30 epochs, Adam, LR decay 0.8 after 3 epochs, early stopping patience 5.
- Fix seeds (torch, numpy, random, cudnn deterministic). Run 3 seeds: 2021, 2022, 2023. Report mean +/- std.
- Log every run to `results/runs.csv` with: run_id, model, dataset, horizon, seed, key flags, MSE, MAE, MASE, epoch_time_s, params, peak_mem_mb.

## Scope
Core datasets: ETTh1, ETTm1, Weather. Horizons 96/192/336/720.
Stretch: Electricity (321 channels). Skip Traffic and Solar-Energy unless everything else is done.

## Architecture (what the model does)
Encoder: split lookback into n = L/w non-overlapping segments -> Linear(w -> d) + ReLU -> single-layer GRU -> final hidden state h_n.
Decoder (PMF): repeat h_n m = H/w times, pair each with a positional embedding pe = concat(rp, cp) where rp = relative position of the future segment and cp = channel identity (both learnable, each dim d/2), run all in parallel through the SAME GRU cell -> Dropout -> Linear(d -> w) -> reshape to H.
Normalization: subtract last input value before encoding, add back after decoding; RevIN for high-shift datasets.
Loss: MSE.
Key point: the model consumes only X. No timestamps anywhere.

## Stage 1 - reconstruction (do first, then freeze)
1. Reproduce Table II numbers for the core datasets.
2. Add baselines the paper lacks, on identical windows/splits/scaling:
   - naive (repeat last value)
   - seasonal naive (period 24 for ETTh, 96 for ETTm, 144 for Weather)
   - optional single linear layer
3. Add MASE alongside the paper's MSE/MAE.
4. Reproduce the authors' own ablations: segment-length sweep including w=1 (Fig. 6), PMF vs RMF (Fig. 7), RNN cell variants via scripts/SegRNN/ablation/rnn_variants.sh (Fig. 8).
5. Verify the drop_last bug fix is present in data_provider/data_factory.py and exp/exp_main.py.

## Stage 2 - improvements (new code goes in models/, keep the original intact)
A. FEATURES: extend pe to concat(rp, cp, tp), where tp comes from learnable nn.Embedding tables indexed by the real timestamp of each future segment (hour-of-day 24, day-of-week 7, month 12). Future timestamps are known in advance, so this is not leakage. First check whether data_provider already returns batch_x_mark / batch_y_mark; if so this is mostly wiring.
B. ARCHITECTURE: bidirectional GRU encoder. Concat forward+backward final states (2d) and project back to d before the decoder, since the decoder shares the cell. Not leakage: the whole lookback window is past data.
   Secondary ablation: replace Linear(w -> d) + ReLU with Conv1d over each segment. Higher risk - ISMRNN (arXiv 2407.10768) reports conv hurting on all datasets except Weather.
C. EFFICIENCY: sweep d_model in {512, 256, 128, 64}, reporting MSE/MAE alongside params, epoch time, and peak memory.
D. OPTIONAL: quantile head (P10/P50/P90) with pinball loss instead of MSE; report empirical coverage vs nominal. Use P50 as the point forecast.

## Working style I want
- Before editing, show me the diff plan and which file you'll touch.
- Small, reviewable changes. New variants as separate model files, never overwriting models/SegRNN.py.
- Every experiment must be re-runnable from a single shell command saved under scripts/.
- If a result looks too good, suspect leakage or a metric bug and say so.

## Environment
There is NO GPU in this local environment. Never launch training here. All runs happen on Google Colab (T4) via notebooks/colab_runner.ipynb, which clones this repo and symlinks the dataset from Google Drive. Locally you read code, write code, and write analysis.
Datasets are never committed: dataset/, checkpoints/, *.csv, __pycache__/ are gitignored. The Colab side gets code changes with `git pull`, so anything I need to run must be pushed first.
New behaviour goes behind flags (e.g. --use_time_emb, --bidirectional, --seg_embed conv) with a matching script under scripts/SegRNN/ in the existing style, so the plain reconstruction always stays runnable for honest comparison.

## Related work I must cite (novelty honesty)
None of these modify SegRNN, but the report must acknowledge them, and I must not claim more novelty than I have.
- CycleNet (NeurIPS 2024 Spotlight, same lab as SegRNN): learnable recurrent cycles model periodicity, backbone predicts the residual; plug-and-play, demonstrated on PatchTST and iTransformer. Closest in spirit to improvement A.
- GLAFF (NeurIPS 2024): timestamps modelled separately (attention mapper + robust denormalizer + adaptive combiner) as a model-agnostic plugin.
- D2Vformer (arXiv 2409.11024): Date2Vec time-position embedding using both input and future date matrices; makes the same leakage-free argument I use.
- IndexNet (arXiv 2509.23813): learnable embedding sets per temporal field (24 hour-of-day vectors, 7 day-of-week, etc). Mechanically the closest to improvement A.
- ISMRNN / MSegRNN (arXiv 2407.10768): implicit segmentation via dual linear projections, residual encoder, Mamba preprocessing. Attacks the same component as improvement B's secondary conv variant.
- TQNet (ICML 2025, same lab): periodically shifted learnable attention queries. It already claims SegRNN's stated future work on inter-channel modelling, so do NOT propose "fix the CP encoding".
- P-sLSTM (AAAI 2025): patching plus channel independence on sLSTM.
Improvement A is not novel as a technique. The contribution is testing it inside SegRNN's positional embedding, which is untested. State that plainly in the report.

## Layout
- CLAUDE.md            this file
- docs/                assignment, lecture PDF, paper PDF, segrnn_project_plan.md, notes
- models/SegRNN.py     the model; annotate first, extend behind flags
- exp/exp_main.py      train/val/test loop
- data_provider/       loaders, splits, scaling  <- check whether batch_x_mark is produced then dropped
- layers/              RevIN etc
- scripts/SegRNN/      per-dataset run scripts = the paper's hyperparameters
- notebooks/           colab_runner.ipynb
- results/runs.csv     experiment log
- dataset/             gitignored; Drive symlink on Colab
