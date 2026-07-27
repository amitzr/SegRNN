# Stage 2 report content — five follow-up architecture ideas (draft)

Draft content for five more independent Stage 2 strands, alongside
`docs/stage2_report_draft.md`, `docs/stage2_efficiency_draft.md`,
`docs/stage2_revin_attn_draft.md`, `docs/stage2_new_strands_draft.md`,
and `docs/stage2_architecture_variants_draft.md`. Sources:
`notebooks/colab_runner_stage2c.ipynb`, `models/SegRNNUnshared.py`,
`models/SegRNNPoolContext.py`, `models/SegRNNWeightTied.py`,
`models/SegRNNLinearShortcut.py`, `models/SegRNNLayerNorm.py`.

Unlike the earlier batches, these five weren't picked from a fixed list
of lecture-adjacent techniques — they were chosen specifically to follow
up on results already in hand (the surprising `SegRNNBidir` outcome, the
confound between bottleneck-removal and added capacity in `SegRNNAttn`)
or to extend the one pattern that's actually worked twice now
(`d_model=256`, Huber loss: reduce/restructure rather than add). Each
tested independently against the same `RECON_BASELINE`, not stacked with
any other strand.

---

## Strand 15: un-shared decode cell (`SegRNNUnshared`)

**What changed:** the encoder stays unidirectional, identical to
`models/SegRNN.py`. Only the decode step gets its own independently-
learned GRU cell instead of reusing the encoder's.

**Why:** `SegRNNBidir` (`docs/stage2_architecture_variants_draft.md`
strand 11) changed two things simultaneously — bidirectional encoding
*and*, as a forced consequence, an un-shared decode cell — and came back
"around the same" as the reconstruction, a genuine surprise (it was
predicted to be the strand most likely to regress clearly, being the
heaviest by parameter count). This strand isolates the un-sharing
variable on its own. If it's also neutral, un-sharing (not
bidirectionality) is likely what's absorbing the extra capacity without
cost. If it regresses on its own, that would instead suggest
bidirectional context was doing real work in `SegRNNBidir`, compensating
for the added decode capacity.

**Results:** pending — run notebook Part 1.

---

## Strand 16: parameter-free pooling context (`SegRNNPoolContext`)

**What changed:** identical to `SegRNNAttn`'s encoder (keeps all `n`
per-segment states, not just `h_n`), but instead of learned attention
(query/key/value projections), a parameter-free mean-pool (or max-pool,
`--pool_type`) collapses those states into one context vector, added
directly to `h_n`. Zero new learnable parameters versus plain `SegRNN` —
unlike `SegRNNAttn`, `SegRNNRocket`, or `SegRNNFFT`, all of which added a
projection layer on top of their feature/context source.

**Why:** `SegRNNAttn` regressed at every horizon
(`docs/stage2_revin_attn_draft.md` strand 4), but it's never been clear
whether that's because giving the decoder more than `h_n` doesn't help
this data, or because the added Q/K/V parameters repeated the same
"added capacity hurts" pattern as everything else. This strand re-asks
the bottleneck-removal question at (essentially) zero added capacity,
separating the two explanations.

**Results:** pending — run notebook Part 2.

---

## Strand 17: weight-tied value embedding / predict head (`SegRNNWeightTied`)

**What changed:** the value-embedding weight (`Linear(seg_len ->
d_model)`) and the predict head's weight (`Linear(d_model -> seg_len)`)
are exact shape-transposes of each other; this strand ties them to one
shared parameter (a "tied autoencoder" pattern), removing
`seg_len * d_model` independent parameters. Biases stay independent.

**Why:** the two genuine wins in this project so far (`d_model=256`,
Huber loss) both *reduce* something rather than add it. This applies the
same direction structurally — enforce parameter sharing by construction,
rather than picking a smaller `d_model` — a genuinely different
mechanism for the same underlying "less is more" finding, untested until
now.

**Results:** pending — run notebook Part 3.

---

## Strand 18: DLinear-style linear shortcut (`SegRNNLinearShortcut`)

**What changed:** a second, parallel `Linear(seq_len -> pred_len)` path
runs directly on the normalized input, bypassing the GRU/segment/PMF
machinery entirely, and is blended with the RNN path's output via a
learned per-channel gate (`sigmoid`, initialized at 0 so the blend starts
at an even 50/50).

**Why:** the paper's own text (Section V-B2) notes that in the univariate
setting, "the lightweight DLinear outperforms the more complex
Transformer-based models such as PatchTST and iTransformer" — simplicity
already wins inside the paper's *own* results, not just in this
project's Stage 2 findings. Blending rather than replacing lets the
model learn, per channel, how much to trust a plain linear view of the
series versus SegRNN's recurrent view.

**Results:** pending — run notebook Part 4.

---

## Strand 19: LayerNorm after value embedding (`SegRNNLayerNorm`)

**What changed:** `nn.LayerNorm(d_model)` appended after the value
embedding's `Linear + ReLU`. There is currently no normalization anywhere
between the segment projection and the GRU encoder.

**Why:** deliberately not in the same family as the strands that added
capacity/information and regressed — `LayerNorm(d_model)` adds `2 *
d_model` parameters (1,024 at `d_model=512`, negligible next to the
GRU's ~1.6M) and gives the model no new information to accommodate. It's
an optimization/conditioning aid, closer in spirit to the loss-function
strand (a training-dynamics change) than to attention/ROCKET/FFT/conv
embedding (all information-injection changes).

**Results:** pending — run notebook Part 5.

---

## Discussion

To be written once all five strands have real results. Stated priors,
before running anything:
- Weight tying and LayerNorm are the two strands most aligned with what
  has actually worked in this project (structural reduction; a
  non-informational training aid) — the closest things to expecting a
  positive or neutral result rather than a regression.
- Pool context and unshared-decode are diagnostic strands as much as
  improvement candidates — their value is in what they reveal about
  *why* `SegRNNAttn` and `SegRNNBidir` behaved the way they did, not just
  whether they beat the reconstruction outright.
- Linear shortcut is the least predictable of the five: it's not a
  capacity reduction (adds `seq_len * pred_len` parameters, non-trivial
  at long horizons) but also isn't a repeat of the "inject more into
  `h_n`" pattern that has failed repeatedly — it's a structurally
  different kind of addition (a whole separate simple mechanism, blended
  in) that doesn't have a close analog among strands tested so far.
