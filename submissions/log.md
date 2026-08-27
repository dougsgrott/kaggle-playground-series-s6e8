# Submission log (append-only)

Budget: **10 per day**, **2** final selections. Deadline 2026-08-31 23:59 UTC.

**Rules**
- Every row records the artifact, the nested-CV OOF it came from, and the public LB it got — the
  pair is what fits the CV→LB line, and that line is the only reason to spend a submission.
- Public LB deltas below **~7e-05** (correlated submissions) or **~1.3e-03** (uncorrelated) are not
  measurable. Rank movement is not evidence.
- Kaggle keeps the best public score, so resubmitting an identical-quality file improves the
  displayed rank with probability ≈ 0.486 for free. Do not read that as progress.

| # | date | tag | artifact | OOF | public LB | offset | final pick | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-26 | `xgb_baseline` | `oof/{oof,test}_xgb_baseline.npy` | 0.964869 | **0.96640** | +0.001531 | — | ref 55801067 · issue 004 · ~rank 1169/3022 |
| 2 | 2026-08-26 | `xgb_features` | `oof/{oof,test}_xgb_features.npy` | 0.968616 | **0.96989** | +0.001274 | — | ref 55804397 · issue 007 · ~rank **685/3022** (top 22.7%) |

**8 submissions remaining today.**

## The CV→LB line

Two points now, so there is a slope for the first time.

| # | CV | LB | offset |
|---|---|---|---|
| 1 | 0.964869 | 0.96640 | +0.001531 |
| 2 | 0.968616 | 0.96989 | +0.001274 |

**The corpus line predicted submission 2 at 0.96982; it scored 0.96989 — a residual of +7.2e-05**,
inside the ~7e-05 the public split can resolve between correlated submissions. Two independent
predictions now, both within 1e-04. The line transfers.

### Our own slope, and why it is not yet a replacement

Fitting our two points gives **−0.0686 ± 0.0264** against the corpus's **−0.1139** — 1.7 standard
errors apart, so not a contradiction, but not a confirmation either. The standard error comes
straight from the ~7e-05 resolution on each offset divided by the 0.0037 CV span between the
points; a *third* point at higher CV is what shrinks it, and the span is what does the work.

Where the two disagree in practice, at the CV levels that matter for selection:

| CV | LB, our slope | LB, corpus slope | spread |
|---|---|---|---|
| 0.9700 | 0.97118 | 0.97104 | 1.4e-04 |
| 0.9705 | 0.97164 | 0.97149 | 1.5e-04 |
| 0.9710 | 0.97211 | 0.97193 | 1.8e-04 |

The disagreement is ~1.5e-04 — larger than the noise floor, so it is worth resolving, but small
relative to the gap between a good stack and a bad one. **Use the corpus slope as the prior until a
third point lands**; it is fitted on eight observations rather than two, and it is the more
conservative of the two (it predicts *lower* LB, so it will not flatter a submission).

Next point wanted at **CV ≈ 0.970+, on a different day** — day separation is not superstition here,
it is the only way to avoid fitting a slope to two runs of the same pipeline on the same afternoon.

## Final selections

| slot | artifact | rationale |
|---|---|---|
| A | _(pending)_ | Our honest nested-CV stack — the bet that a real CV edge survives 237,042 private rows |
| B | _(pending)_ | Plateau consensus — the bet that the herd is right |

Rationale must be written **before** the final public standing is seen.
