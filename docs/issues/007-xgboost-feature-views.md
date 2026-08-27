# 007 — XGBoost GPU members across feature views

**Status:** done 2026-08-26
**Phase:** 2 — the member zoo
**Do first in Phase 2.** XGBoost GPU is the only family here that trains a full 5-fold in minutes
rather than hours, so it is both the cheapest member to produce and the right vehicle for feature-
view diversity. Every slower family (005 CatBoost, 006 LightGBM) is priced against what this
establishes.

## Why this one goes first

Two reasons, and the second is the binding one.

1. **`xgb_features` has never been run properly.** Issue 002 measured it on a deliberately cheap
   fixed-round config (lr 0.10 × 900, no early stopping) chosen to make 16 ablation arms
   affordable. It reached pooled OOF **0.967910** that way. The real member — early stopping, lr
   0.035, cap 6000 — has not been trained, and its OOF is what every stack weight downstream will
   be fitted against.
2. **The CV→LB line rests on a single point.** Submission 1 sits at CV 0.964869 → LB 0.96640.
   A line through one point has no slope, and [`../../submissions/log.md`](../../submissions/log.md)
   is explicit that the slope is the part that matters for final selection. `xgb_features` should
   land near CV 0.968, which is exactly the second point wanted — and it is wanted *on a different
   day* from the third, so this submission is time-sensitive in a way the modelling is not.

## Scope

- [x] Train `xgb_features` on the frozen folds with early stopping; export the positional OOF
      contract and submit it. Record OOF, LB, offset, and the residual against the corpus line.
- [x] Build **3–4 additional XGBoost views** that differ by *feature set*, not by hyperparameter
      seed — see the view list below.
- [x] Rank-correlation matrix over every member in `oof/`, written to
      [`../experiments.md`](../experiments.md).
- [x] Apply the admission rule below and record which views earn a place and which do not,
      including the ones that do not.

### The views

Chosen so each removes or replaces a *mechanism*, since the admission rule kills members that
differ only by seed. Expected rank correlation to the flagship in brackets.

| view | blocks | why it might decorrelate |
|---|---|---|
| `xgb_features` | all six + `te` | the flagship |
| `xgb_no_te` | all six, no `te` | drops the target-reading mechanism entirely; leans on `lattice`+`freq` for value separation |
| `xgb_te_only` | `raw` + `budget` + `te` | the opposite bet — target encoding without the lattice |
| `xgb_deep` | all six + `te`, `max_depth` 10, higher `min_child_weight` | different bias/variance point on identical features (**expected to fail admission** — recorded anyway, since "same algorithm, different params → weight 0.000" is a corpus claim worth testing once on our own data) |

### Admission rule (from the roadmap, measured in the corpus)

A member must be **decorrelated AND comparably strong**. Rank correlation above ~0.99 to an
existing member, or more than ~0.006 AUC weaker, and it earns weight 0.000. There is a visible
strength cliff near solo OOF 0.966 — contribution tracks solo OOF more than it tracks
decorrelation.

## Outcome

Five members on the frozen folds. Flagship submitted: **OOF 0.968616 → LB 0.96989**, rank
**~685 / 3,022** (top 22.7%), up from ~1,169. Predicted 0.96982, residual **+7.2e-05**.
Full table: [`../experiments.md`](../experiments.md#007--xgboost-feature-views-and-the-admission-rule).

| member | solo OOF | ρ to flagship | blend contribution |
|---|---:|---:|---:|
| `xgb_features` | **0.968616** | — | anchor |
| `xgb_te_only` | 0.968596 | 0.9980 | +0.000135 |
| `xgb_deep` | 0.968531 | 0.9976 | +0.000100 |
| `xgb_no_te` | 0.967691 | 0.9928 | **+0.000228** |
| `xgb_baseline` | 0.964869 | 0.9865 | +0.000108 |
| **all five, nested** | **0.968948** | — | **+0.000334** |

## What it bought beyond a number

1. **The admission rule is backwards at this stack size, and the rule is now amended.** Rank
   correlation rank-orders blend contribution *in reverse*: ρ 0.9928 buys +0.000228, ρ 0.9865 buys
   +0.000108. Applying the rule as written would have kept the two least useful members and
   thrown away the most useful. The likely reconciliation is stack size — at 70 members a
   0.99-correlated addition is genuinely redundant, at five it still reduces variance — and the
   saturation is visible (the 5th member adds only +0.000016). `CLAUDE.md` and the roadmap now say
   *do not discard on ρ alone yet, re-test as the zoo grows*. The strength half stands.
2. **Feature views inside one algorithm do not decorrelate, so this line of work is finished.**
   Every pair sits at ρ 0.9859–0.9980; dropping target encoding entirely reaches only 0.9928.
   Diversity must come from other families — the corpus's GBDT→NN figure is ρ 0.974 at weight 0.22,
   a regime unreachable with trees. **Do not build more XGBoost views**; the marginal one is worth
   ~1e-05.
3. **Issue 002's headline was an ordering artifact, and this issue resolved it.** `xgb_te_only`
   prices the lattice/TE pair in the opposite order: dropping the lattice costs 0.00002 when `te`
   is present, dropping `te` costs 0.000925 when the lattice is. So TE dominates, and issue 002's
   "+0.001609, 16× the corpus" was credit for shared signal handed to whichever block was measured
   first. This *reconciles* us with the corpus, which measured the lattice with TE already in
   place. The general lesson: **a single-block delta from a cumulative ablation is a property of
   the block AND the order**, never of the block alone.
4. **The CV→LB line has a slope for the first time** — −0.0686 ± 0.0264 against the corpus's
   −0.1139, 1.7 se apart. Not enough to replace the corpus prior, which is fitted on eight points
   and is the more conservative of the two. Detail and the recommendation:
   [`../../submissions/log.md`](../../submissions/log.md).
5. **`xgb_deep` was run to fail and did not.** Depth 10 cost 0.000085, not the corpus's −0.0011,
   and it earned +0.000100 in the blend rather than the predicted weight 0.000. Cheap, recorded,
   and a second corpus claim that does not transfer.

## Built along the way

| what | where |
|---|---|
| `_xgb_view` — one parameterised builder, four registered views | `src/s6e8/models.py` |
| rank-correlation matrix, admission verdicts, nested-CV blend check | `scripts/member_matrix.py` |

## Get right the first time

1. **Early stopping selects on the fold it is scored on.** Issue 004 flagged this: the resulting
   OOF is mildly optimistic and is *not* a clean nested estimate. It stays comparable to published
   numbers, which is why it is used, but Phase 3 must not fit stack weights on a naive full-OOF
   basis on top of it — that compounds two optimisms. Nested CV at the stacker is already
   non-negotiable #3.
2. **Raise the round cap.** The starter's 3000 barely bound (best iterations 2521–2818, issue 004).
   `xgb_features` is set to 6000; if best iterations approach that, raise it rather than reporting
   a truncated model.
3. **σ is feature-set specific** (issue 002). Do not compare these members against the 0.00011 gate
   from issue 003 — for member-vs-member comparisons on *different* feature sets the relevant
   quantity is σ_partition-like, and for admission the relevant quantity is not a gate at all but
   the rank correlation and the strength cliff.
4. **Check the submission against the line before spending a slot.** Predicted offset is
   `0.00150 − 0.1139·(CV − 0.9660)`. A miss beyond ~1e-4 is evidence of a pipeline problem, not of
   a better model.

## Expected

`xgb_features` at CV ≈ 0.9680–0.9685 → LB ≈ 0.9692–0.9697. That would be roughly **+0.003** on the
board over submission 1 and would put the CV→LB fit on two well-separated points.

The corpus's best documented *single* LightGBM is CV 0.96862 → LB 0.96990, so a flagship XGBoost
landing near 0.968 is in the right neighbourhood but not expected to beat it — LightGBM with a
large `max_bin` is the stronger single family here (issue 006).

## Exit criterion — met

Five members in `oof/`, rank-correlation matrix and admission verdicts in
[`../experiments.md`](../experiments.md), second CV→LB point in
[`../../submissions/log.md`](../../submissions/log.md), and the admission rule itself tested
rather than assumed.

## Next

**Not more XGBoost views** — that is settled at ~1e-05 per marginal member. In priority order:

1. **Issue 006, LightGBM + `max_bin` sweep.** The strongest single family in the corpus
   (CV 0.96862 → LB 0.96990) and the one place `max_bin` is still unpriced against our feature set.
   Expect the Phase-0 +0.00142 to shrink, since `lattice`/`freq`/`te` already do value separation.
2. **The NN members** (Lookup-Transformer, RealMLP) — the only route to real decorrelation, per
   finding 2. Higher value than any remaining tree work.
3. Issue 005 (CatBoost), blocked by [016](016-catboost-gpu-eval-metric.md).

A Phase-3 stack of the current five is already worth **+0.000334 nested** and is a submission
waiting to happen — but the CV→LB line wants its third point on a **different day**, so that is a
tomorrow job, not a tonight one.
