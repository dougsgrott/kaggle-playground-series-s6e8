# 003 — Measure the noise floor locally

**Status:** open
**Phase:** 1
**Estimate:** ~1 h
**Do second** — after 004, before 002.

## Why this goes before the feature module

Every row in `docs/experiments.md` currently divides its delta by **0.00005**, which is the
*corpus's* noise floor, not ours. Until it is measured locally the `× floor` column is borrowed,
and so is every "this beat the floor" judgement made against it.

The cost dropped sharply once XGBoost GPU started working. Repeated 5-fold across 3 partition
seeds is **15 fits at ~27 s** at 400 rounds — roughly 7 minutes of compute, with the rest of the
hour going to the harness.

## Scope

- [ ] `src/s6e8/cv.py::repeated_cv(model_fn, X, y, seeds=(42, 2024, 7))` — full 5-fold per seed,
      returning the per-seed pooled OOF AUCs
- [ ] Run it on **XGBoost GPU**, not LightGBM
- [ ] Report `std(ddof=1)` of the per-seed **means** as the noise floor
- [ ] Also report the per-fold range within one run, and the ratio between the two
- [ ] Write the number into `src/s6e8/config.py::NOISE_FLOOR`, `docs/data-notes.md`, and the rules
      block at the top of `docs/experiments.md`
- [ ] Recompute the existing `× floor` values in `docs/experiments.md` against the local number

## Method notes

- **Use XGBoost GPU.** At 27 s/fold it is 4–8× cheaper than any other family here and is immune to
  the CPU contention that makes LightGBM timings load-dependent. A noise floor measured on a
  contended CPU path would carry that variance into the estimate.
- The partition seed is what varies — `StratifiedKFold(5, shuffle=True, random_state=seed)` — not
  the model seed. Model reseeding is a different, smaller quantity.
- **The per-fold spread inside a single run is not the noise floor** and overstates it by roughly
  an order of magnitude. Report both so the distinction is visible in the artifact, not just in
  the prose.
- This does not touch `data/folds.npy`. That file stays frozen at seed 42; the extra seeds live
  only inside this measurement.

## Expected

Somewhere near the corpus value of **0.00005**, with the per-fold range ~10× larger. If the local
floor comes out much bigger, every published delta in `docs/experiments.md` needs rereading before
Phase 2 commits to anything.

## Exit criterion

A local noise-floor number in `config.py` and the docs, and no remaining `× floor` value computed
from the borrowed constant.

## Next

Issue 002 (feature module).
