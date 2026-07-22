# SegRNN architecture notes: code ↔ paper mapping

Source read: `models/SegRNN.py` (the active `Model` class, lines 9–135) and
its call sites in `exp/exp_main.py`. A second, commented-out "concise
version" implementation also lives at the bottom of `models/SegRNN.py`
(lines 137–203) — it is dead code but useful as a reference because it
strips away every ablation branch and matches the paper's main
configuration (GRU, PMF, channel-id on) almost line for line.

## Paper → code, block by block

### 1. Segment partition
Paper: split each length-`seq_len` (`s`) series into `n = s / w` segments of
length `w` (`seg_len`).

Code: `models/SegRNN.py:29` — `self.seg_num_x = self.seq_len // self.seg_len`
computes `n`. The actual split happens in `forward` at
`models/SegRNN.py:84`: `x.reshape(-1, self.seg_num_x, self.seg_len)`, after
the input has been permuted to `b,c,s` and flattened so batch and channel
share a leading dimension (`bc,n,w`).

### 2. `Linear(w→d)` + ReLU
Code: `models/SegRNN.py:32-35`, `self.valueEmbedding`. Applied at
`models/SegRNN.py:84`, producing `bc,n,d`. Matches the paper exactly.

### 3. Single-layer GRU → `h_n`
Code: `models/SegRNN.py:40-42` builds `self.rnn = nn.GRU(d_model, d_model,
num_layers=1, batch_first=True)` when `rnn_type == "gru"`. Encoding happens
at `models/SegRNN.py:90`: `_, hn = self.rnn(x)`, discarding the per-step
outputs and keeping only the final hidden state `h_n` (shape `1,bc,d`).
Matches the paper.

Caveat: `rnn_type` is actually a 3-way switch (`rnn` / `gru` / `lstm`,
asserted at `models/SegRNN.py:26`) — RNN and LSTM are ablation options not
in the paper's described architecture, which specifies GRU.

### 4. PMF decode
This is the `dec_way == "pmf"` branch, `models/SegRNN.py:105-127`
(the alternative `"rmf"` branch, `93-104`, is a recurrent multi-step decoder
that is a separate ablation not covered by the paper excerpt given — see
flags below).

- **`pe = concat(rp, cp)`** — built at `models/SegRNN.py:106-116`.
  `self.pos_emb` (`seg_num_y, d/2`) is the relative-position embedding
  `rp`, `self.channel_emb` (`enc_in, d/2`) is the channel-position
  embedding `cp`. They're broadcast to every channel/position pair and
  concatenated on the last dim to form `pe` of shape `bcm,1,d`
  (`m = seg_num_y = pred_len // seg_len`).
- **Repeat `h_n` `m` times** — `models/SegRNN.py:125`:
  `hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)`, turning
  `1,bc,d` into `1,bcm,d`.
- **Parallel pass through the *same* GRU cell** — `models/SegRNN.py:125`:
  `_, hy = self.rnn(pos_emb, hn.repeat(...))`. This reuses `self.rnn`,
  the identical module instance used for encoding — not a separate decoder
  RNN — which is exactly the "same GRU cell" behavior the paper describes.
  Because `pos_emb` carries no sequential dependency between the `m`
  positions, this single call decodes all `m` future segments in parallel
  rather than autoregressively.
- **Dropout → `Linear(d→w)` → reshape** — `self.predict` at
  `models/SegRNN.py:62-65` (`Dropout(dropout)` then `Linear(d_model,
  seg_len)`), applied at `models/SegRNN.py:127`:
  `self.predict(hy).view(-1, self.enc_in, self.pred_len)`.

## Deviations / extras not in the paper excerpt

1. **Normalization step precedes segment partition** (`models/SegRNN.py:76-81`)
   and isn't mentioned in the given paper section, which starts at segment
   partition. Two modes exist:
   - default (`revin=0`, the default in `run_longExp.py`'s argparse):
     subtract the last observed value per-channel (`seq_last`), add it back
     after decoding (`models/SegRNN.py:133`). This is simple last-value
     normalization, consistent with common linear-forecaster baselines.
   - optional (`revin=1`): full RevIN (`layers/RevIN.py`), an ablation not
     described in the paper excerpt.
2. **`dec_way="rmf"`** (`models/SegRNN.py:93-104`) is a recurrent,
   autoregressive alternative to PMF: it predicts one segment at a time and
   feeds each prediction back into the GRU as the next input. Not part of
   the paper description given, and not what the reference script
   (`scripts/SegRNN/etth1.sh`) uses (`--dec_way pmf`).
3. **`channel_id=0`** disables the channel embedding entirely
   (`models/SegRNN.py:59-60`, only `pos_emb` at full `d_model` width, no
   concatenation with `channel_emb`). The paper's `pe = concat(rp, cp)`
   assumes `channel_id=1`, which is also the reference script's default.
4. **`rnn_type in {rnn, lstm}`** — ablations on top of the paper's GRU
   choice, adding an LSTM cell-state branch (`cn`) threaded alongside `hn`
   throughout `forward`.

## exp/exp_main.py call site

`exp/exp_main.py` dispatches on model name substring
(`{'Linear', 'SegRNN', 'TST'}`, e.g. lines 78-79, 156-157, 263-264,
358-359): for SegRNN it calls `self.model(batch_x)` only — no timestamp
features, no decoder input, no teacher-forcing target — consistent with
`Model.forward(self, x)` taking a single positional tensor. See
`docs/timefeatures_audit.md` for what happens to `batch_x_mark` /
`batch_y_mark` on this path.
