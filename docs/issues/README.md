# Issues

One file per unit of work: `NNN-slug.md`. Pick up work here; tick scope boxes and set `Status:`
in place. Never delete a closed issue — it is the record of what was tried.

The full task index lives in [`../ROADMAP.md`](../ROADMAP.md#task-index). This table is the
**execution order**, which is not the numbering order.

## Next up

| order | # | title | est. | status |
|---:|---|---|---|---|
| 1 | 002 | [`features.py`, the measured-positive set](002-feature-module.md) | the real work | **next** |

**002 now has a real gate.** Issue 003 measured **σ_delta = 0.000055**, so a feature block must
clear **2σ = 0.00011** to have measured anything. Three of its planned blocks — the decimal
lattice, the explicit `__missing__` level, 10 encoding folds — sit at +0.0001 in the corpus and
are therefore *under* the gate: build them, but do not book them as gains without the multi-seed
protocol. The top four blocks (+0.0003 to +0.0023) are clear by 5×–42×.

Also from 003, and useful throughout Phase 2: **the model seed carries twice the variance of the
partition seed** (0.000039 vs 0.000019), so precision on a marginal ablation is bought by
averaging model seeds — σ_delta falls as √n — not by repeating CV across partitions.

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
| 003 | [Measure the noise floor locally](003-noise-floor.md) | done 2026-08-26 · 55 fits · σ_delta **0.000055**, gate **0.00011** |
