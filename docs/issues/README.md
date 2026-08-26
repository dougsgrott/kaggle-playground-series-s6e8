# Issues

One file per unit of work: `NNN-slug.md`. Pick up work here; tick scope boxes and set `Status:`
in place. Never delete a closed issue — it is the record of what was tried.

The full task index lives in [`../ROADMAP.md`](../ROADMAP.md#task-index). This table is the
**execution order**, which is not the numbering order.

## Next up

| order | # | title | est. | status |
|---:|---|---|---|---|
| 1 | 003 | [Measure the noise floor locally](003-noise-floor.md) | ~1 h | **next** |
| 2 | 002 | [`features.py`, the measured-positive set](002-feature-module.md) | the real work | open |

**003 runs before 002** because every `× floor` value in [`../experiments.md`](../experiments.md)
is still divided by the corpus constant rather than our own measurement. Issue 004 made the case
concrete: its fold range was **0.001629**, about **33×** the corpus floor of 0.00005. Reading
per-fold spread as uncertainty would hide every real gain the feature work is trying to find.

The harness 003 needs now exists — `src/s6e8/cv.py::run_member` and the `xgb_baseline` member — so
the remaining work is `repeated_cv` over three partition seeds.

## In parallel

| # | title | when | status |
|---|---|---|---|
| 014 | [Research track: the smoothed boundary band](014-boundary-band-research.md) | day 2–3, while Phase 2 members train | open |

## Loose ends — clear early

| # | title | status |
|---|---|---|
| 015 | [Re-run the benchmark on a quiet machine](015-rerun-benchmark-quiet.md) | open |
| 016 | [CatBoost GPU: drop `eval_metric="AUC"`](016-catboost-gpu-eval-metric.md) | open (blocks 005) |

## Closed

| # | title | outcome |
|---|---|---|
| 001 | [Phase 0 — environment, data, frozen folds](001-phase-0-ground-truth.md) | done 2026-08-26 · gate PASS, 16 checks |
| 004 | [Baseline member and the first submission](004-baseline-first-submission.md) | done 2026-08-26 · OOF 0.964869 → LB **0.96640** |
