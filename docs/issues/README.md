# Issues

One file per unit of work: `NNN-slug.md`. Pick up work here; tick scope boxes and set `Status:`
in place. Never delete a closed issue — it is the record of what was tried.

The full task index lives in [`../ROADMAP.md`](../ROADMAP.md#task-index). This table is the
**execution order**, which is not the numbering order.

## Next up

| order | # | title | est. | status |
|---:|---|---|---|---|
| 1 | 006 | LightGBM + `max_bin` sweep — the strongest single family | ~3 h | **next** |
| 2 | — | NN members: Lookup-Transformer, RealMLP | the diversity | open |
| 3 | 005 | CatBoost native categoricals (blocked by [016](016-catboost-gpu-eval-metric.md)) | ~2 h | open |

**Stop building XGBoost views.** Issue 007 settled it: five members span ρ 0.9859–0.9980 and the
marginal one is worth ~1e-05. Real decorrelation needs other families — the corpus's GBDT→NN figure
is ρ 0.974 at blend weight 0.22, a regime trees cannot reach. That makes the **NN members higher
value than any remaining tree work**, ahead of their roadmap position.

Two rules changed under measurement and Phase 2 should carry the amended versions:

- **Do not discard a member on rank correlation alone.** Contribution came out rank-ordered
  *backwards* by ρ over our own five members. Amended in `CLAUDE.md`; the strength half stands.
- **A cumulative-ablation delta belongs to the block *and the order*.** The lattice/TE pair
  inverts depending on which is measured first. Price a substitute pair in both orders or report
  neither number as the block's value.

A Phase-3 stack of the current five is worth **+0.000334 nested** (0.968948) and is the obvious
next submission — but the CV→LB line wants its third point on a **different day**.

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
| 007 | [XGBoost GPU members across feature views](007-xgboost-feature-views.md) | done 2026-08-26 · 5 members · LB **0.96989**, rank ~685/3022 · admission rule inverted |
