# Stage 2 report content — Yeo-Johnson, AIC/BIC, ensembling, ROCKET (draft)

Draft content for four more independent Stage 2 strands, alongside the
calendar-feature work (`docs/stage2_report_draft.md`), the efficiency
sweep (`docs/stage2_efficiency_draft.md`), and RevIN/Attention
(`docs/stage2_revin_attn_draft.md`). Sources: notebook Parts 0a–0d,
`data_provider/data_loader.py`, `models/SegRNNRocket.py`,
`results/runs.csv`.

Each strand is tested independently against the same reconstruction
baseline used throughout (`RECON_BASELINE` in the notebook), not stacked
with each other or with the calendar-feature/RevIN/Attention strands —
same reasoning as `docs/stage2_revin_attn_draft.md`'s Discussion: stacking
an untested idea onto an already-negative strand would make it impossible
to attribute the result to either one.

---

## Summary: all eight Stage 2 strands

Assignment categories are the instructor's own list
(`docs/segrnn_project_plan.md`'s "Coverage of the instructor's improvement
categories" table). "Paper tried this?" is about the SegRNN paper
specifically (`docs/SegRNN_paper.pdf`), not the wider literature.

| # | Strand | Assignment category | Result | Paper tried this? | Lecture-grounded? |
|---|---|---|---|---|---|
| 1 | Calendar features (`SegRNNTime`, 3 attempts) | Add stronger time-series features | **Failed** — regressed at all 4 horizons, seed-confirmed real, not noise | No — the model consumes only raw values, no timestamps anywhere in the paper's design; related work (CycleNet, GLAFF, D2Vformer, IndexNet — `CLAUDE.md`'s novelty-honesty list) tries similar ideas on other backbones, never on SegRNN | Partial — `docs/DL_for_TS.pdf`'s embedding-layer slide covers embedding categorical/cyclical features generally, not this specific use |
| 2 | RevIN normalization | Improve preprocessing and data quality | **Failed** — worse at all 4 horizons (+3.1% to +4.6% MSE) | Yes, but not here — the paper uses RevIN, applied specifically to datasets with "severe distribution shift" (e.g. Traffic), not as the ETTh1 default; this strand deliberately tests it somewhere the paper itself doesn't recommend it | Yes — `docs/DL_for_TS.pdf`'s Embedding Layer slide gives the exact z-score formula |
| 3 | Attention over encoder states (`SegRNNAttn`) | Improve ML or deep-learning architecture | **Failed** — worse at all 4 horizons (+0.5% to +5.1% MSE) | No — SegRNN's whole design point is *avoiding* attention/Transformer machinery for efficiency; attention is what the paper's baselines (Informer/Autoformer/PatchTST/Transformer) use, not what SegRNN itself does | Yes — `docs/DL_for_TS.pdf`'s Attention slide, "information bottleneck" diagnosis |
| 4 | `d_model` efficiency sweep (256/128/64) | Improve computational efficiency | **Succeeded** — `d_model=256` keeps 99.65% of accuracy at 25.5% of the parameters at H≤336 (H=720 costs more, +4.8%) | No — the paper doesn't ablate `d_model` for SegRNN itself (only PatchTST's default width appears, in the Table VI runtime comparison) | Not explicitly — written before the lecture PDFs existed; retroactively reinforced by strand 6's AIC/BIC read of the same data |
| 5 | Yeo-Johnson power transform | Improve preprocessing and data quality | **Best result overall, with a real trade-off** — MSE improves 3.6-7.4% at every horizon (the only strand that beats the paper's own headline metric), MAE worsens 4.1-6.7% at every horizon | No — the paper's preprocessing is plain `StandardScaler`, fit on train only; no power transform | Yes — `docs/Pre-precessing.pdf`'s preprocessing slide |
| 6 | AIC/BIC post-hoc analysis of the `d_model` sweep | Improve or tune the forecasting model | **Free analysis, not a trained model** — both criteria reject `d_model=512` at every horizon and go further than strand 4's own informal "knee" reading, favoring 64 or 128 everywhere | No — the paper uses validation loss + early stopping for model selection, not a formal information criterion | Yes — `docs/Time-Series Forecasting.pdf`'s model-selection slide |
| 7 | Multi-seed prediction ensembling | Doesn't cleanly fit any instructor category (closest: improve or tune the forecasting model) | **Mostly neutral** — a borderline-noise small win at H=336, clearly noise at H=720 | No — the paper's own project plan flags this as a gap in the paper itself: "main tables are single runs," no multi-seed reporting | No — general technique, not covered by either lecture PDF; flagged as such when proposed |
| 8 | ROCKET-style random-convolution features (`SegRNNRocket`) | Add stronger time-series features | **Failed, worst regression of any strand** — worse at 3/4 horizons, H=336's +8.7% MSE is the largest single-horizon regression tested in this project | No — ROCKET is a classification-era technique; no textbook or paper precedent for using it inside a forecaster | Yes, as a feature-engineering technique — `docs/Pre-precessing.pdf`; using it inside a forecaster rather than a classifier is an untaught adaptation, flagged as such when proposed |

**Overall tally:** one clean success (`d_model` sweep), one best-of-project
result with a genuine trade-off (Yeo-Johnson — the only strand that beats
the paper's own headline MSE), one zero-cost analysis that reinforces the
efficiency direction rather than standing on its own (AIC/BIC), one
neutral strand (ensembling), and four failures (calendar features, RevIN,
attention, ROCKET). The four failures share a structural signature laid
out in this document's and `docs/stage2_revin_attn_draft.md`'s Discussion
sections: every strand that *added* information or architectural richness
regressed; every strand that *reduced* capacity or reshaped the input
distribution without adding new information either succeeded outright or
produced a genuine, honestly-reported trade-off.

---

## Strand 5: Yeo-Johnson power transform (preprocessing)

**What changed:** `data_provider/data_loader.py`'s `Dataset_ETT_hour`
gained a `power_transform` flag (`run_longExp.py --power_transform 1`, off
by default). When on, it fits `sklearn.preprocessing.PowerTransformer
(method='yeo-johnson', standardize=True)` on the train split only, in
place of the plain `StandardScaler` — a drop-in swap, not an addition (the
transform's `standardize=True` already zero-means/unit-variances the
transformed values, so nothing downstream changes shape-wise). With the
flag off, behavior is byte-identical to the plain reconstruction.

**Why it should help:** `docs/Pre-precessing.pdf`'s preprocessing slide
lists Yeo-Johnson as the standard fix for heteroscedasticity and
non-normality in a feature — unlike Box-Cox, it's defined for negative
values, which matters here since ETTh1's `OT` and load channels take
negative values. `StandardScaler` only recenters/rescales; it doesn't
address skew in the *distribution* of each channel, which the paper's
Table I load statistics suggest is plausible for at least a couple of
ETTh1's seven channels.

**Results:**

| Horizon | Reconstruction MSE | Yeo-Johnson MSE | Δ | Reconstruction MAE | Yeo-Johnson MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | 0.3250 | -7.4% | 0.3925 | 0.4085 | +4.1% |
| 192 | 0.3925 | 0.3640 | -7.3% | 0.4142 | 0.4368 | +5.5% |
| 336 | 0.4233 | 0.4038 | -4.6% | 0.4327 | 0.4616 | +6.7% |
| 720 | 0.4657 | 0.4489 | -3.6% | 0.4719 | 0.4926 | +4.4% |

**Reading the result:** a genuinely different shape of result from every
other strand so far — MSE and MAE disagree in *sign*, consistently, at
every horizon. This hasn't happened in any of strands 1-4 (calendar
features, RevIN, attention, `d_model` sweep all moved MSE and MAE
together). MSE improves by a real margin (3.6-7.4%, largest at the
shorter horizons) while MAE consistently worsens (4.1-6.7%).

The likely explanation is exactly what Yeo-Johnson is designed to do:
compress the scale of extreme values to stabilize variance. MSE weights
squared error, so it is dominated by the model's worst few predictions —
shrinking those outliers' scale before training plausibly lets the model
fit them far better, which shows up as a large MSE improvement. MAE
weights every error equally regardless of magnitude, so it's more
sensitive to a shift in the *typical* prediction's accuracy — if
compressing extreme values costs the model a little precision on
ordinary-magnitude points (e.g., a coarser effective resolution near the
distribution's center after the transform), that shows up as MAE getting
worse even while the outlier-dominated MSE improves. This is a testable
follow-up (e.g., look at the residual distribution/error histogram,
not just the two aggregate metrics), not confirmed here.

Practically: since the paper's headline metric is MSE (Table II), this is
the first Stage 2 strand tested so far that actually beats the
reconstruction on the paper's own primary comparison point — a real,
positive result, alongside the caveat that it costs MAE. Worth reporting
as a genuine trade-off rather than a clean win, and worth checking whether
it holds up under the same seed-variance scrutiny Part 6 gave the
calendar-feature finding (not yet done for this strand).

---

## Strand 6: AIC/BIC post-hoc analysis of the `d_model` sweep

**What changed:** nothing — this is a zero-cost re-analysis of Part 7's
already-completed `d_model` sweep (`docs/stage2_efficiency_draft.md`), no
new training. `docs/Time-Series Forecasting.pdf`'s model-selection slide
gives the formal criteria AIC = n·ln(MSE) + 2k and BIC = n·ln(MSE) +
k·ln(n) (Gaussian-error regression form) — the quantitative version of
"prefer the simplest model that explains the data well," the same
principle behind the same deck's Theta-method note that simple methods
often outperform complex ones. `n` = test windows × horizon × channels for
each split; `k` = SegRNN's exact analytical parameter count
(`scripts/plot_efficiency_sweep.py`'s formula).

**Why this is worth doing:** the efficiency sweep's headline (`d_model`
=256 keeps 99.65% of accuracy at 25.5% of the parameters at H=336) was
read informally, off a "knee" in an accuracy-vs-params plot. AIC/BIC put a
number on the same trade-off and can disagree with that informal read —
worth knowing either way, and free to check.

**Results:**

| Horizon | d_model | MSE | Params (k) | n | AIC | BIC |
|---|---|---|---|---|---|---|
| 96 | 512 | 0.3510 | 1,603,864 | 1,871,520 | 1,248,565 | 21,204,260 |
| 96 | 256 | 0.3557 | 408,728 | 1,871,520 | -1,117,074 | 3,968,427 |
| 96 | 128 | 0.3677 | 106,072 | 1,871,520 | -1,660,289 | -340,514 |
| 96 | 64 | 0.3848 | 28,472 | 1,871,520 | **-1,730,417** | **-1,376,161** |
| 192 | 512 | 0.3925 | 1,604,888 | 3,614,016 | -170,354 | 20,854,208 |
| 192 | 256 | 0.3956 | 409,240 | 3,614,016 | -2,532,984 | 2,828,195 |
| 192 | 128 | 0.4007 | 106,328 | 3,614,016 | **-3,092,514** | -1,699,582 |
| 192 | 64 | 0.4219 | 28,600 | 3,614,016 | -3,061,649 | **-2,686,979** |
| 336 | 512 | 0.4233 | 1,606,424 | 5,985,840 | -1,933,069 | 19,922,180 |
| 336 | 256 | 0.4248 | 410,008 | 5,985,840 | -4,304,682 | 1,273,439 |
| 336 | 128 | 0.4304 | 106,712 | 5,985,840 | **-4,832,880** | -3,381,073 |
| 336 | 64 | 0.4445 | 28,792 | 5,985,840 | -4,795,766 | **-4,404,054** |
| 720 | 512 | 0.4657 | 1,610,520 | 10,891,440 | -5,102,134 | 17,772,867 |
| 720 | 256 | 0.4881 | 412,056 | 10,891,440 | -6,987,610 | -1,134,977 |
| 720 | 128 | 0.4915 | 107,736 | 10,891,440 | -7,520,645 | -5,990,418 |
| 720 | 64 | 0.4950 | 29,304 | 10,891,440 | **-7,600,226** | **-7,184,007** |

(bold = the criterion's pick for that horizon; lower AIC/BIC is better)

Which criterion prefers which `d_model`, per horizon:

| Horizon | MSE picks | AIC picks | BIC picks |
|---|---|---|---|
| 96 | 512 | 64 | 64 |
| 192 | 512 | 128 | 64 |
| 336 | 512 | 128 | 64 |
| 720 | 512 | 64 | 64 |

**Reading the result:** AIC and BIC both reject the paper's default
`d_model=512` outright — not once, at any horizon, does either criterion
prefer it, despite it having the lowest MSE every time. More strikingly,
**neither criterion ever picks `d_model=256`** either, the value
`docs/stage2_efficiency_draft.md` read informally as the practical
"knee" in the accuracy-vs-params curve. AIC/BIC push further than that
informal read: they land on 64 or 128 every time. BIC is the more
extreme of the two — it picks the smallest available option (`64`) at
*every* horizon, because its penalty (`k·ln(n)`) is far harsher than
AIC's (`2k`) once `n` is in the millions (`ln(n) ≈ 14–16` here). AIC is
more discriminating: it splits between 64 (H=96, H=720) and 128 (H=192,
H=336), essentially trading off the small remaining accuracy gap against
the parameter cost at each horizon individually rather than defaulting
to the floor every time.

**A caveat that matters more than the exact break-even points:** `n`
here is total scalar predictions (test windows × horizon × channels),
which overstates the number of *independent* observations — adjacent
windows overlap almost entirely (shifted by one timestep out of
`seq_len=720`), and the seven channels aren't independent draws either.
A more defensible `n` would be smaller. Since AIC/BIC's accuracy term
scales with `n` while the parameter penalty doesn't, a smaller, more
honest `n` would shrink the accuracy term's influence *relative to* the
parameter penalty — pushing the verdict even further toward small
`d_model`, not back toward 512 or 256. So the inflated `n` used here, if
anything, has been generous to the larger models; the qualitative
conclusion (formally, complexity is punished hard) is robust to that
criticism even though the exact horizon-by-horizon 64-vs-128 split
shouldn't be read too precisely.

---

## Strand 7: Multi-seed prediction ensembling

**What changed:** `exp/exp_main.py`'s `test()` now optionally saves raw
`pred.npy`/`true.npy` (`--save_preds 1`, off by default, no behavior
change otherwise). The notebook reruns the reconstruction at 3 seeds
(2021, 2022, 2024) with `--save_preds 1`, averages the three seeds'
predictions element-wise, and scores the *averaged prediction* — not the
same thing as Part 6's seed-variance check, which averages the *metrics*
across seeds rather than the predictions.

**Why it's here:** not lecture-grounded (unlike the other three strands in
this document) — flagged honestly as a general technique rather than one
from `docs/Pre-precessing.pdf` or `docs/Time-Series Forecasting.pdf`.
Cheap to test given Part 6 already established the seed-to-seed spread at
these two horizons.

**Results:**

| Horizon | Reconstruction MSE | Ensemble MSE | Δ | Reconstruction MAE | Ensemble MAE | Δ | seed MSEs | seed std |
|---|---|---|---|---|---|---|---|---|
| 336 | 0.4233 | 0.4222 | -0.26% | 0.4327 | 0.4320 | -0.16% | 0.4250, 0.4237, 0.4232 | 0.0009 |
| 720 | 0.4657 | 0.4659 | +0.04% | 0.4719 | 0.4725 | +0.13% | 0.4659, 0.4744, 0.4656 | 0.0050 |

**Reading the result:** small and horizon-dependent. At H=336 the delta
(-0.0011 MSE) is close to the same magnitude as the seed-to-seed std
(0.0009) — using Part 6's own heuristic (`|mean delta| vs std`), this
lands right at the boundary rather than clearly on either side; a
plausible small real effect, not a strong one. At H=720 the delta
(+0.0002) is an order of magnitude smaller than the seed std (0.0050) —
clearly noise, no effect either way. In both cases, averaging
*predictions* beats averaging *metrics*: the ensemble MSE is better than
the mean-of-seeds MSE at both horizons (0.4222 vs 0.4240 mean at H=336;
0.4659 vs 0.4686 mean at H=720) — expected, since scoring an averaged
prediction removes each individual model's variance-driven error before
squaring, while averaging already-squared per-seed errors does not get
that benefit (Jensen's inequality). So the technique does what it's
supposed to relative to the naive alternative; it just doesn't move the
needle much relative to a single reconstruction run here, and what
movement there is falls inside or right at the edge of ordinary seed
noise rather than being a clear win.

---

## Strand 8: ROCKET-style random-convolutional features (`SegRNNRocket`)

**What changed:** `models/SegRNNRocket.py` adds a fixed (untrained, buffer
-registered) bank of random 1D convolution kernels over the raw
(last-value-normalized) lookback window, per channel — Max and PPV
(proportion of positive values) pooling per kernel, the two statistics the
ROCKET paper uses. A trainable `Linear` projects this feature vector to
`d_model` and adds it to `h_n` before decoding:
`h_n_aug = h_n + rocket_proj(rocket_features)`. Everything else (segment
partition, value embedding, GRU encoding, PMF decode, normalization) is
identical to `models/SegRNN.py`.

**Why it should help, and the caveat:** `docs/Pre-precessing.pdf`'s
feature-engineering slide lists ROCKET (random convolutional kernels) as a
window-based feature technique. ROCKET is a classification-era method —
there's no textbook precedent for using it inside a forecaster, so this is
an honest adaptation, not a documented one; it's the most speculative of
the four strands in this document. It's also a direct repeat of the
"inject extra information into `h_n`" pattern that all three
calendar-feature attempts used and that regressed every time
(`docs/stage2_report_draft.md`) — testing it with a completely different,
non-calendar information source (features of the raw series itself) is an
honest, independent check of whether that earlier finding is
calendar-specific or a more general property of this architecture/dataset.

**Results:**

| Horizon | Reconstruction MSE | ROCKET MSE | Δ | Reconstruction MAE | ROCKET MAE | Δ |
|---|---|---|---|---|---|---|
| 96 | 0.3510 | 0.3693 | +5.2% | 0.3925 | 0.4029 | +2.6% |
| 192 | 0.3925 | 0.3975 | +1.3% | 0.4142 | 0.4203 | +1.5% |
| 336 | 0.4233 | 0.4603 | +8.7% | 0.4327 | 0.4611 | +6.6% |
| 720 | 0.4657 | 0.4650 | -0.15% | 0.4719 | 0.4785 | +1.4% |

**Reading the result:** worse at three of four horizons, essentially flat
at the fourth (H=720's -0.15% MSE is noise-scale, but its MAE still
worsens +1.4%, so even the "best" horizon isn't a real win). H=336 is the
worst single-horizon regression of any Stage 2 strand tested across this
whole project (+8.7% MSE, +6.6% MAE — bigger than Attention's +5.1% at
the same horizon, bigger than RevIN's worst point of +4.6%). This matches
the prediction made before running it (see Discussion below, and this
strand's own rationale above): ROCKET repeats the exact "inject extra
information into `h_n`" pattern that calendar features and attention both
already used, with a third, unrelated information source now producing
the same outcome. That's no longer a coincidence limited to calendar
data — three independent information sources injected into the same
bottleneck (calendar timestamps, attention-derived context, and now
fixed random-convolution features of the raw window) have all made
things worse. The pattern looks like it's about *where* the information
goes, not *what* the information is.

---

## Discussion

**Checking the prediction made before running anything** (stated above,
before results existed, precisely so it would be falsifiable): three
parts held, one didn't.

- ROCKET regresses — **confirmed**, and more decisively than expected
  (worst single-horizon regression of any strand in the whole project).
- Ensembling is a coin flip — **confirmed**: a small, borderline-noise
  effect at one horizon, clean noise at the other.
- AIC/BIC "confirms or sharpens" the `d_model=256` finding — **wrong**.
  It doesn't sharpen 256, it bypasses it entirely: both criteria prefer
  64 or 128 at every horizon, never 256. The prediction assumed AIC/BIC
  would land near the same answer as the informal "knee" read; instead
  they disagree with it, in the same direction (favor less capacity) but
  further than expected.
- Yeo-Johnson regresses or is neutral — **wrong on MSE, right on MAE**.
  It's the one strand this document got backwards on the metric that
  matters most for comparing against the paper.

**How this fits the wider Stage 2 picture.** `docs/stage2_revin_attn_draft.md`
found a clean pattern across the first four strands: every attempt to
give the model *more* (calendar features, RevIN's richer normalization,
attention's bottleneck removal) made things worse; only *reducing*
capacity (`d_model` sweep) helped. ROCKET extends that pattern to a third
independent information source injected the same way (into `h_n`) — same
outcome, reinforcing that the failure mode is structural (*where* the
information goes), not specific to calendar data or to any one
information source. AIC/BIC, applied to the one strand that *did* help,
turns out to agree with the "less is more" reading even more strongly
than the original analysis did — it doesn't just ratify `d_model=256` as
a reasonable compromise, it argues the paper's whole capacity budget at
`d_model=512` is hard to justify formally, at any horizon.

Ensembling and Yeo-Johnson are the two strands that don't fit that
add/reduce axis at all — neither changes what the model can represent.
Ensembling reduces prediction *variance* without touching the model, and
behaved exactly like a variance-reduction technique should: a small,
mostly noise-scale effect, present but not dramatic given only 3 seeds.
Yeo-Johnson changes the *distribution* of the input, not the model's
capacity or the information available to it — and it's the only strand
in this entire project (across both this document and
`docs/stage2_revin_attn_draft.md`) that improves the paper's headline
metric (MSE) at every horizon. That its MAE gets worse at the same time
is a real trade-off, not a contradiction: it's a different kind of lever
than everything else tested (preprocessing, not architecture or
capacity), and the two metrics respond to it differently because they
weight errors differently. Worth flagging as the strongest positive
result in the report, with the caveat that (unlike the calendar-feature
finding) it hasn't yet been checked for seed variance.

**Limitations:** Yeo-Johnson, ROCKET, and the AIC/BIC analysis are each
single-seed (2024) or, for AIC/BIC, derived from the single-seed sweep in
Part 7 — none have the seed-variance confirmation Part 6 gave the
calendar-feature finding. Ensembling is the only strand here that used
multiple seeds directly, and even then only 3. All four strands, like
every other one in this project, are ETTh1-only.
