# Data pipeline audit

Source read: `data_provider/data_loader.py`, `data_provider/data_factory.py`,
`utils/timefeatures.py`. Covers loading, cleaning, datetime parsing,
resampling, missing-value handling, scaling, temporal split validity, and
feature construction — the checklist required for the assignment's Stage 1
reconstruction write-up.

## 1. Loading

`Dataset_ETT_hour.__read_data__` (`data_provider/data_loader.py:43-46`):
`pd.read_csv(root_path/data_path)`. For ETTh1: columns are
`[date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]` — one datetime column plus
6 covariates + 1 target (`OT`), all numeric, no header transformation.
`Dataset_Custom` (used for `--data custom`, e.g. Weather/Electricity) does
the same read but additionally reorders columns to `[date] + others +
[target]` (`data_provider/data_loader.py:229-232`).

## 2. Cleaning / missing-value handling

**None.** There is no `dropna`, interpolation, gap-filling, or outlier
handling anywhere in `data_provider/data_loader.py`. The loader assumes the
CSV is already a complete, gap-free, fixed-frequency series. This holds for
the ETT benchmark files (they're pre-cleaned research datasets with no
missing timestamps), but it is a real gap if you point this pipeline at a
messier dataset — flag this explicitly in the report, and note it's a
legitimate Stage 2 "improve preprocessing and data quality" opportunity
(the assignment brief lists this as an example improvement).

## 3. Datetime parsing

`df_stamp['date'] = pd.to_datetime(df_stamp.date)` (e.g.
`data_provider/data_loader.py:67`). No timezone handling, no format string
— relies on pandas' inference, which works for ETT's `YYYY-MM-DD HH:MM:SS`
format but isn't validated defensively.

## 4. Resampling

**None.** `freq` (e.g. `'h'` for ETTh1) is passed in via CLI flag and used
only to pick which calendar features to compute (§7) — it is never used to
actually resample or validate the series' cadence. The pipeline trusts the
CLI-declared frequency matches the CSV's true spacing.

## 5. Scaling

`StandardScaler` (sklearn), fit **only on the train split's rows**
(`train_data = df_data[border1s[0]:border2s[0]]`,
`data_provider/data_loader.py:60-61`), then applied via `.transform()` to
the full series (train+val+test). This is correct practice — no
train/val/test leakage into the scaler's mean/variance. Per-channel scaling
(one mean/std per column), inverted at evaluation time via
`inverse_transform` for metric computation in the original units.

## 6. Temporal split protocol — leakage check

**`Dataset_ETT_hour`** (ETTh1/ETTh2), `data_provider/data_loader.py:48-49`:
```python
border1s = [0, 12*30*24 - seq_len, 12*30*24 + 4*30*24 - seq_len]
border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
```
Fixed, chronological, non-overlapping index ranges: train = first 12
"months" (30-day months, 8640 hours), val = next 4 months, test = the
8 months after that. The `- seq_len` offset on val/test's start lets their
first prediction window look back into the immediately preceding split for
input context — this is standard and **not leakage**, since it only reads
already-past rows relative to that window's own prediction target, and
those rows were already visible to training as raw series values (not
model updates).

**`Dataset_Custom`** (generic/custom datasets),
`data_provider/data_loader.py:234-238`: same mechanism, but split by
proportion (70% train / 20% test / remainder val) instead of fixed month
counts.

Both splits are **strictly chronological by row index** — no shuffling of
the underlying series. The `DataLoader`'s `shuffle_flag=True` for training
(`data_provider/data_factory.py:30`) only shuffles the *order in which
already-built sliding windows are drawn per batch* — each window itself is
still a contiguous, correctly-ordered past→future slice, so this does not
violate temporal causality. **Verdict: the split is valid** — no future
information reaches training, scaling, or (implicitly) hyperparameter
choice through this mechanism.

One caveat worth stating in the report: hyperparameters (e.g. `seg_len`,
`d_model`, learning rate) are the paper's published defaults, not tuned
by us against this val/test split — so there's no risk of us having
leaked test performance into hyperparameter search, but it also means our
reconstruction doesn't independently validate the paper's tuning.

## 7. Feature construction — calendar/time features

Controlled by `--embed` (default `'timeF'` per `run_longExp.py`'s
argparse) → `data_factory.py:16`: `timeenc = 1` when `embed == 'timeF'`.

- **`timeenc=1` (default path):** `utils/timefeatures.py`'s `time_features()`
  is called (`data_provider/data_loader.py:75-76`). For hourly data
  (`freq='h'`) this returns, per timestamp:
  `[HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]` (`utils/timefeatures.py:92`),
  each continuous-valued and normalized to `[-0.5, 0.5]`
  (e.g. `HourOfDay = hour/23.0 - 0.5`, `utils/timefeatures.py:34-38`).
  Shape: `(window_len, 4)`.
- **`timeenc=0` (alternative, `--embed fixed` or `--embed learned`):**
  raw integer calendar columns are built directly
  (`data_provider/data_loader.py:68-73`): `month, day, weekday, hour`
  (plus `minute//15` for minute-frequency data) — meant for an embedding
  *table* lookup rather than continuous input, used by the Transformer-family
  models in this repo, not by SegRNN.

**Where these features go on the SegRNN path:** `exp/exp_main.py` computes
`batch_x_mark`/`batch_y_mark` for every model (train/vali/test loops each
load and `.to(device)` them), but the dispatch condition
`if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'})`
calls `self.model(batch_x)` only — no mark tensors passed
(confirmed at `exp/exp_main.py:79,87,157,171,264,272,359,367`, and in
`docs/architecture_notes.md`'s note on the `exp_main.py` call site). So for
SegRNN specifically, calendar features are computed and moved to GPU every
batch, then **immediately discarded** — the model never sees them.

**Implication for a calendar-embedding improvement:** this is mostly
*wiring*, not building from scratch. `data_stamp` is already computed with
the right values, shape, and normalization by the existing loader; the
work is (a) changing `SegRNN.forward` to accept and use `x_mark`, (b)
deciding an injection point (e.g. concatenate encoded time features into
the segment embedding before the GRU, or fold them into the existing
`pos_emb`/`channel_emb` PMF decode stream), and (c) updating
`exp_main.py`'s SegRNN branch to actually pass `batch_x_mark`/`batch_y_mark`
through. No new datetime feature-extraction logic is needed.
