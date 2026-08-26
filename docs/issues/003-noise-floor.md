# 003 — Measure the noise floor locally

**Status:** done 2026-08-26
**Phase:** 1
**Estimate:** ~1 h
**Do second** — after 004, before 002.  ·  Actual cost: 31.6 min of compute, 55 fits.

## Why this goes before the feature module

Every row in `docs/experiments.md` currently divides its delta by **0.00005**, which is the
*corpus's* noise floor, not ours. Until it is measured locally the `× floor` column is borrowed,
and so is every "this beat the floor" judgement made against it.

The cost dropped sharply once XGBoost GPU started working. Repeated 5-fold across 3 partition
seeds is **15 fits at ~27 s** at 400 rounds — roughly 7 minutes of compute, with the rest of the
hour going to the harness.

## Scope

- [x] `src/s6e8/cv.py::repeated_cv(model_fn, X, y, seeds=(42, 2024, 7))` — full 5-fold per seed,
      returning the per-seed pooled OOF AUCs
- [x] Run it on **XGBoost GPU**, not LightGBM
- [x] Report `std(ddof=1)` of the per-seed **means** as the noise floor
- [x] Also report the per-fold range within one run, and the ratio between the two
- [x] Write the number into `src/s6e8/config.py::NOISE_FLOOR`, `docs/data-notes.md`, and the rules
      block at the top of `docs/experiments.md`
- [x] Recompute the existing `× floor` values in `docs/experiments.md` against the local number

## Method notes

- **Use XGBoost GPU.** At 27 s/fold it is 4–8× cheaper than any other family here and is immune to
  the CPU contention that makes LightGBM timings load-dependent. A noise floor measured on a
  contended CPU path would carry that variance into the estimate.
- The partition seed is what varies — `StratifiedKFold(5, shuffle=True, random_state=seed)` — not
  the model seed. Model reseeding is a different, smaller quantity.
  > **Wrong, as measured.** Model reseeding is the *larger* quantity here (0.0000391 vs
  > 0.0000193), and it is the one an ablation is actually exposed to. The run swept both.
  > See Outcome, point 2. Left in place as the record of what was assumed going in.
- **The per-fold spread inside a single run is not the noise floor** and overstates it by roughly
  an order of magnitude. Report both so the distinction is visible in the artifact, not just in
  the prose.
- This does not touch `data/folds.npy`. That file stays frozen at seed 42; the extra seeds live
  only inside this measurement.

## Expected

Somewhere near the corpus value of **0.00005**, with the per-fold range ~10× larger. If the local
floor comes out much bigger, every published delta in `docs/experiments.md` needs rereading before
Phase 2 commits to anything.

## Outcome

**55 XGBoost-GPU fits — 6 partition seeds × 6 model seeds — in 31.6 min.** Full cell-level record:
[`../noise_floor.json`](../noise_floor.json). Runner: `scripts/noise_floor.py`.

| quantity | measured | corpus |
|---|---:|---:|
| σ_partition (pooled AUC across partition seeds) | **0.0000193** | 0.00005 |
| σ_partition (of the fold *means*, as scoped) | 0.0000195 | — |
| σ_model (across model seeds, frozen partition) | 0.0000391 | ±0.00004 |
| **σ_delta = √2·σ_model** — the operative floor | **0.0000552** | — |
| max null delta actually observed | 0.000101 | — |
| mean per-fold range | 0.001215 (**22× σ_delta**) | ~10× |

The probe model pooled to 0.964371, within 0.0005 of the `xgb_baseline` member — same regime, a
third of the cost (lr 0.10 × 900 fixed rounds vs 0.035 × ~2700 with early stopping).

## What it bought beyond a number

1. **The scope asked for the wrong quantity, and the run measured both.** σ_partition answers "how
   far does the CV number move when the split changes" — but no experiment in this repo changes
   the split. They all hold it at seed 42 and vary the config, so the null they must beat is the
   spread of an A/B *delta* with both arms re-seeded. That is σ_delta, and it is **2.9× larger**
   than σ_partition. Gating on σ_partition would have passed roughly-null results as real.
2. **Model seed beats partition seed 2:1** (0.0000391 vs 0.0000193). Pooling 691,369 rows makes
   the split nearly irrelevant; `subsample`/`colsample_bytree` redrawing in five models does not.
   The practical consequence contradicts the method note in the scope above: **repeated CV across
   partitions is the wrong way to buy precision.** Averaging model seeds is — σ_delta falls as √n,
   to 0.000032 at 3 seeds per arm. Issue 002 should budget seeds, not partitions.
3. **The per-fold range is a property of the split, not an error bar.** On the frozen partition
   fold 3 is easier by **+0.00097** and fold 0 harder by **−0.00061**, reproduced across all six
   model seeds. That is why partition 42's fold range (0.00142–0.00170) is *disjoint* from every
   other partition's (0.00050–0.00116), and it fully explains issue 004's alarming 0.001629.
   Corollary: the fold-0 probes in rows 001a–001e read ~0.0006 low against a pooled number.
4. **The corpus constant was right by luck.** Its 0.00005 is within 9% of σ_delta — but it was
   labelled the partition quantity, and ours is 0.0000193. The number survived, the attribution
   did not. It also gates at 1× where the honest gate is 2×, which retires every "build" verdict
   in its table at or below +0.0001 (see [`../experiments.md`](../experiments.md)).

## Built along the way

| what | where |
|---|---|
| `make_fold_assignment(y, seed, n_splits)` — one definition of the partition rule | `src/s6e8/data.py` |
| `repeated_cv`, `CVCell`, `RepeatedCVResult`, `FitScore` | `src/s6e8/cv.py` |
| the runner, with a `--rounds` smoke-test override | `scripts/noise_floor.py` |
| `SIGMA_PARTITION`, `SIGMA_MODEL`, `SIGMA_DELTA`, `NOISE_FLOOR`, `FOLD_RANGE_TYPICAL` | `src/s6e8/config.py` |

`scripts/make_folds.py` now calls the shared rule and still reproduces sha256 `9571f18b…`, so the
frozen partition is provably untouched. Alternate partitions are built in memory only; `FitScore`
skips test inference, saving a 296,302-row predict per fit.

## Exit criterion — met

Local numbers in `config.py`, [`../data-notes.md`](../data-notes.md) and
[`../experiments.md`](../experiments.md); the two `× floor` values recomputed (62×→56×, 26× from
28×), and no remaining figure divided by the borrowed constant.

## Amended by issue 002 — the gate is not portable

Issue 002 re-measured σ_model on seven different feature sets and found it spans **7×**:
0.000005 at the raw 12 columns, peaking at 0.000032 around 36 columns, back to 0.000017 at 63,
against the **0.000039** measured here on the 34-column one-hot starter block.

So **σ_delta = 0.000055 and the 0.00011 gate belong to the feature set they were measured on**,
not to the box. Applying them uniformly would have been far too strict at the raw end and too
loose in the middle. It is also *not* monotonic in column count — seed sensitivity peaks where
many columns are comparably useful and `colsample_bytree` has real choices, and collapses when
the model has either too little to choose from or one dominant feature to lock onto.

Everything else here stands: the two-quantity distinction, model seed beating partition seed, the
per-fold range being a property of the split, and the multi-seed protocol. What changes is that an
ablation must carry **its own** σ rather than borrow this one. `scripts/ablate_features.py` reports
one per arm and computes each comparison's null sd as `sqrt(σa²/na + σb²/nb)`.

## Next

Issue 002 (feature module) — **done 2026-08-26**, +0.003903 over raw.
