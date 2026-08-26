# Issues

One file per unit of work: `NNN-slug.md`. Pick up work here; tick scope boxes and set `Status:`
in place. Never delete a closed issue — it is the record of what was tried.

The full task index lives in [`../ROADMAP.md`](../ROADMAP.md#task-index). This table is the
**execution order**, which is not the numbering order.

## Next up

| order | # | title | est. | status |
|---:|---|---|---|---|
| 1 | 007 | Phase 2 — XGBoost GPU members across feature views | the main effort | **next** |

`xgb_features` is registered and ablated (+0.003903 over raw, OOF 0.967910 on the cheap
fixed-round config). The first job in Phase 2 is a real OOF from it with early stopping, and a
submission — the CV→LB line still rests on a **single** point at CV 0.9649, and a second one near
CV 0.968 is what makes the slope usable for final selection.

Two findings from 002 that change how Phase 2 measures things:

- **σ_model is a property of the feature set, not the box.** It spans 7× across the sets measured
  (0.000005 → 0.000032) and is not monotonic in column count. The 0.00011 gate from issue 003 is
  not portable; every ablation carries its own null sd. See
  [`003`](003-noise-floor.md#amended-by-issue-002--the-gate-is-not-portable).
- **Published deltas are substitutes and must never be summed.** `lattice`, `freq`, `te` and
  `max_bin` all do value separation. Measured last, `te` priced at 0.32× its published value.
  This is why 006 must sweep `max_bin` *after* the encodings, not before.

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
| 003 | [Measure the noise floor locally](003-noise-floor.md) | done 2026-08-26 · 55 fits · σ_delta **0.000055** (feature-set specific — amended by 002) |
| 002 | [`features.py`, the measured-positive set](002-feature-module.md) | done 2026-08-26 · **+0.003903** over raw · 6 blocks ship, 1 UNRESOLVED |
