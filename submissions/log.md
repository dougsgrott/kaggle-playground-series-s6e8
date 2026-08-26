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
| 1 | 2026-08-26 | `xgb_baseline` | `oof/{oof,test}_xgb_baseline.npy` | 0.964869 | **0.96640** | +0.001531 | — | ref 55801067 · issue 004 · ~rank 1154/2987 |

**9 submissions remaining today.**

## The CV→LB line

The whole reason submission 1 was made before the feature work. Two points define nothing, so this
starts the fit rather than finishing it — but the first point already says something useful:

| CV | LB | offset |
|---|---|---|
| 0.964869 | 0.96640 | **+0.001531** |

The corpus line (+0.00150 at CV 0.9660 decaying to +0.00109 at CV 0.9696) predicts **+0.001629**
here. Observed is **+0.001531** — a residual of **−9.8e-05**, which is inside the ~1e-4 range where
a single public score cannot resolve anything anyway.

**So the published line transfers to our pipeline.** Until we have points of our own at higher CV,
use it to sanity-check a submission before spending a slot: a member at CV *x* should land near
`x + 0.00150 − 0.1139·(x − 0.9660)`. A submission that misses that by much more than 1e-4 is
evidence of a pipeline problem, not of a better model.

Next points wanted at **CV ≈ 0.967 and ≈ 0.970**, spread across days — the line's value is its
slope, and two clustered points cannot measure one.

## Final selections

| slot | artifact | rationale |
|---|---|---|
| A | _(pending)_ | Our honest nested-CV stack — the bet that a real CV edge survives 237,042 private rows |
| B | _(pending)_ | Plateau consensus — the bet that the herd is right |

Rationale must be written **before** the final public standing is seen.
