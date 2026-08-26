# 004 — Baseline member and the first submission

**Status:** done (2026-08-26)
**Phase:** 1
**Estimate:** ~30 min
**Done.** Ran before 003 and 002, as planned.

## Why this goes first

It is out of roadmap order deliberately, for two reasons:

1. **The submission budget is a clock, not a pool.** 10 per day with 5 days left, and the CV→LB
   offset is *a line, not a constant* — it decays from +0.00150 at CV 0.9660 to +0.00109 at
   CV 0.9696 (`corr = −0.99`). Fitting that line needs several well-separated CV levels, spread
   over days. A submission not made today is time that cannot be recovered later.
2. **It exercises the whole path** — folds → member → OOF contract → submission CSV → Kaggle —
   while the pipeline is still small enough that a break is obvious. Every later phase assumes
   this path works.

## Scope

- [x] Port `analysis/nb_clean/cdeotte__simple-xgb-starter.py` onto the frozen folds
      (`data/folds.npy`), not its own inline `StratifiedKFold`
- [x] Route it through `src/s6e8/cv.py` so the OOF contract is enforced in one place
- [x] Export `oof/oof_xgb_baseline.npy` (691,369) and `oof/test_xgb_baseline.npy` (296,302) via
      `src/s6e8/data.py::save_member`
- [x] `src/s6e8/submit.py` → `submissions/xgb_baseline.csv`, with the pre-upload assertions
- [x] Submit; record the CV/LB pair in `submissions/log.md` and a row in `docs/experiments.md`

## Notes

- Use `device="cuda"`. `scripts/check_env.py` already asserts xgboost is a CUDA-12 build; the
  member script should assert it too — a 3.x wheel falls back to CPU with a warning, not an error.
- The starter's own feature block (`missing_count`, ratios, shares) is *not* the measured-positive
  set. Port it **as written** anyway: this issue is a path test and a calibration point, not a
  modelling attempt. Issue 002 replaces the features.
- Cap threads before importing xgboost — see `CLAUDE.md` → Environment traps.

## Expected

**~0.9640 OOF / ~0.9655 LB.** A materially different LB means the submission path is wrong, not
that the model is interesting.

## Exit criterion

Met.

## Outcome

| | |
|---|---|
| pooled OOF AUC | **0.964869** |
| public LB | **0.96640** (submission 55801067, ~rank 1154/2987) |
| offset | **+0.001531** |
| per-fold | 0.96423 · 0.96483 · 0.96480 · 0.96585 · 0.96464 |
| fold mean ± sd | 0.964871 ± 0.000600 |
| cost | 8.5 min, ~100 s/fold on the GPU |
| best iterations | 2802, 2521, 2641, 2818, 2764 (cap 3000) |

Expected ~0.9640 OOF / ~0.9655 LB; landed slightly above both. The path works end to end.

### What it bought beyond a number

- **The CV→LB line transfers.** The corpus line predicts +0.001629 at this CV; observed +0.001531,
  a residual of −9.8e-05. That is inside what a single public score can resolve, so the published
  slope is usable as a pre-submission check until we have higher-CV points of our own.
- **The fold range is 0.001629 — about 33× the corpus noise floor.** Concrete confirmation of why
  issue 003 has to measure the floor properly: reading per-fold spread as uncertainty would hide
  every real gain.
- **Early stopping barely bound** (2521–2818 of 3000). The starter's cap is close to limiting;
  future XGBoost members should raise it.
- **The OOF is mildly optimistic** — early stopping picks the iteration on the same fold the OOF is
  scored on. Kept, because it is what the source notebook and the corpus do and it keeps the number
  comparable. Not acceptable for anything whose weights get fitted downstream.

### Built along the way

| file | role |
|---|---|
| `src/s6e8/runtime.py` | thread capping before any OpenMP pool is sized; `assert_xgboost_gpu()` |
| `src/s6e8/cv.py` | `run_member` — the one place the positional OOF contract is enforced |
| `src/s6e8/models.py` | the `REGISTRY`, and `xgb_baseline` ported verbatim |
| `src/s6e8/submit.py` | `write_submission` with the pre-upload assertions |
| `scripts/train_member.py` | `--model <name> [--submit-tag <tag>]` |
| `scripts/make_submission.py` | `--from <name> [--tag] [--rank]` |

The submission assertions were verified against a probe before the real run: id order matches
`sample_submission`, and wrong-length and non-finite inputs are both rejected.

## Next

Issue 003 (noise floor), then 002 (feature module).
