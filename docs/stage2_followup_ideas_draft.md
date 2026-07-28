# Stage 2 report content — follow-up architecture ideas (draft)

Draft content for seven more independent Stage 2 strands (five original,
plus two revised variants built after the originals underperformed),
alongside
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

**Results:** regresses, and the regression **grows with horizon** — H=96
comes back roughly tied with the reconstruction, but the gap widens
steadily through H=720 (exact per-horizon table not yet transcribed
here). This is a different failure shape from `SegRNNBidir`'s flat
"around the same" across all four horizons.

**Reading the result, combined with strand 16 below:** since
`SegRNNBidir` (bidirectional encoding *and* an unshared decode cell) was
flat across horizons, while this strand (unshared decode cell *alone*)
gets worse as horizon grows, the natural read is that bidirectional
encoding's extra context is what was compensating for the unshared
decoder's weakness at longer horizons in `SegRNNBidir` — not that
un-sharing itself is free. Un-sharing alone has a real cost that scales
with how much the decoder has to produce; bidirectional context happens
to offset exactly that cost. A more informative finding than a flat
"neutral" would have been.

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

**Results:**

| Horizon | Reconstruction MSE | Pool Context MSE | Δ | Reconstruction MAE | Pool Context MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | 0.3484 | -0.7% | 0.3925 | 0.3888 | -0.9% |
| 192 | 0.3925 | 0.3854 | -1.8% | 0.4142 | 0.4110 | -0.8% |
| 336 | 0.4233 | 0.4233 | 0.0% | 0.4327 | 0.4371 | +1.0% |
| 720 | 0.4657 | 0.4583 | -1.6% | 0.4719 | 0.4671 | -1.0% |

**Reading the result — the cleanest positive result in the project.**
Improves *both* MSE and MAE at three of four horizons (96, 192, 720);
H=336 is a wash (MSE exactly tied, MAE a small +1.0% cost). This
directly resolves strand 16's own open question: giving the decoder
access to more than `h_n` *does* help this data — `SegRNNAttn`'s
regression was about the added Q/K/V parameters, not the bottleneck-
removal idea itself. And it does so at **zero added parameters** versus
plain `SegRNN` — no projection layer, no learned weights of its own at
all, just a pooling operation. Alongside `d_model=256` and Huber loss,
this is the third genuine win in the project, and by parameter cost, the
cheapest of the three.

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

**Results:** mixed and mostly small — MSE consistently a bit worse across
horizons; MAE roughly tied, with a small improvement at H=720 (exact
per-horizon table not yet transcribed here). Not a clean win like strand
16, but not the clear regressions strands 15/18/19 show either.

**Reading the result:** weaker support for the "structural capacity
reduction" hypothesis than `d_model=256` or Huber loss gave. Tying the
value-embedding and predict weights removes real capacity
(`seg_len * d_model` parameters) but also forces the same representation
to serve two different jobs (turning a segment into a hidden state, and
turning a hidden state back into a segment) — those may not be as
symmetric as the tied-autoencoder framing assumes, which could explain
why this reduction doesn't help the way `d_model=256`'s more uniform
shrink did.

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

**Results:** worse (exact per-horizon table not yet transcribed here).
See the Discussion's follow-up note below for a proposed better-grounded
variant (real trend/seasonal decomposition, and post-hoc prediction
averaging instead of joint-training blend) rather than abandoning the
idea outright.

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

**Results:** worse (exact per-horizon table not yet transcribed here).
See the Discussion's follow-up note below for a proposed repositioned
variant (normalizing the GRU's hidden state instead of the input
embedding) rather than abandoning the idea outright.

---

## Strand 20: LayerNorm on `h_n` instead of the input embedding (`SegRNNLayerNormHidden`)

**What changed:** revised placement of strand 19's idea, which
regressed. Same `nn.LayerNorm(d_model)`, same near-zero parameter cost
(`2 * d_model`), applied to `h_n` immediately after encoding — right
before it initializes the decode step — instead of to the input
embedding right after `Linear + ReLU`.

**Why:** the original placement may have been double-normalizing on top
of the existing last-value subtraction, or disrupting a scale
relationship the GRU had already learned to expect at its input.
Normalizing `h_n` instead targets the actual bottleneck representation —
the same thing pool context (strand 16) and attention (already tested)
both operate on — rather than a step further upstream. This is the
placement a Transformer block would use (normalize hidden states between
sub-layers, not raw input embeddings).

**Results:** pending — run notebook Part 6.

---

## Strand 21: post-hoc ensemble of independently-trained `SegRNN` + `DLinear`

**What changed:** revised version of strand 18 (linear shortcut), which
regressed. Instead of a single plain `Linear(seq_len -> pred_len)`
jointly trained and blended into `SegRNN` via a learned gate, this trains
`SegRNN` and the repo's own `DLinear` (`models/DLinear.py`, already
reconstructed as a Stage 1 baseline candidate) completely independently
— each with `--save_preds 1` — then averages their raw predictions
(not their metrics) and scores the averaged prediction. `DLinear`'s
`--individual` flag already exists in `run_longExp.py` (defaults to 0,
shared weights across channels), so no new CLI flags were needed.

**Why this should fix strand 18's two likely problems:**
1. **A stronger linear model.** Strand 18's shortcut was one flat
   `Linear(seq_len, pred_len)` — a materially weaker model than real
   `DLinear`, which decomposes the input into a moving-average trend and
   a seasonal residual, each with its own linear map. The paper's own
   text (Section V-B2) is about *DLinear specifically* beating
   Transformer baselines in the univariate setting, not "any linear
   layer" — strand 18 tested a weaker proxy for the thing that was
   actually motivating it.
2. **No joint-training interference.** Strand 18's blend gate and the
   RNN path were trained together, so a poorly-initialized or slowly-
   converging gate could have let each path's gradients disrupt the
   other's early training. Training both models fully independently,
   then blending only at prediction time, removes that risk entirely —
   the same "average predictions, not metrics" logic strand 7's seed
   ensembling used, applied across model families instead of seeds.

**Results:** pending — run notebook Part 7.

---

## Discussion

**Checking the prediction made before running anything:** weight tying
and LayerNorm were flagged as "the closest things to expecting a
positive or neutral result" — **wrong for both**; both regressed, while
pool context (framed as a diagnostic strand first, improvement candidate
second) turned out to be this batch's clear win. Diagnostic value of the
un-shared/pooled pair came through as hoped: strand 15 (regresses,
growing with horizon) plus `SegRNNBidir`'s flat result together point at
bidirectional context specifically compensating for the un-shared
decoder's cost, not un-sharing being free on its own. Linear shortcut was
correctly flagged as the least predictable — it regressed, joining
weight tying and LayerNorm rather than pool context.

**Net for this batch: one clear win (pool context, zero added
parameters), one mixed/inconclusive result (weight tying), three
regressions (unshared decode, linear shortcut, LayerNorm).** Combined
with every strand tested across all three documents, pool context is the
strongest single piece of evidence yet that "the encoder's bottleneck
hurts this data" is a real effect independent of implementation cost —
it took three attempts at removing that bottleneck (attention, ROCKET/FFT
injected into `h_n`, and now pooling) to find the one that actually pays
for itself.

**Follow-up variants worth building for the two ideas that "sound right"
but underperformed as implemented (linear shortcut, LayerNorm) — not
abandoning either, revising the mechanism:**

- **Linear shortcut → real DLinear-style decomposition, and/or post-hoc
  blending instead of joint-training blending.** The tested version was a
  single plain `Linear(seq_len -> pred_len)`, not real DLinear (which
  splits the input into a moving-average trend and a seasonal residual,
  each with its own linear map) — a weaker version of the cited
  motivation than the paper's own DLinear baseline. Separately, jointly
  training an in-network blend gate risks both paths pulling each other's
  gradients around early in training, before the gate has learned a
  sensible split. A cleaner test: train `SegRNN` and `DLinear` (already in
  `models/`, a baseline this project already reconstructed) completely
  independently, then average their *predictions* post-hoc — the same
  "average predictions, not metrics" logic strand 7's ensembling used,
  just across model families instead of seeds, avoiding the joint-training
  interference risk entirely.
- **LayerNorm → normalize the GRU's hidden state, not the input
  embedding.** The tested placement (right after `Linear + ReLU`,
  encoder-input side) may be double-normalizing on top of the existing
  last-value subtraction, or disrupting a scale relationship the GRU has
  already learned to expect at its input. Applying `LayerNorm` to `h_n`
  itself, right before decoding — normalizing the bottleneck
  representation the same way a Transformer block normalizes hidden
  states, not the raw input — is a different placement of the same
  near-zero-cost idea, untested here.
