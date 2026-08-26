# 016 — CatBoost GPU: drop `eval_metric="AUC"`

**Status:** open (fixed in `scripts/benchmark.py`; **not** yet in any member script)
**Phase:** loose end — clear before issue 005

## The problem

`eval_metric="AUC"` is **not implemented on CatBoost's GPU backend**. Passing it does not error —
it prints

```
Default metric period is 5 because AUC is/are not implemented for GPU
```

and forces a slow path. A 400-iteration fit on one fold exceeded **500 s** before this was removed;
at 200 iterations without it, the same fit takes **102.8 s**.

Nearly every CatBoost recipe in `analysis/nb_clean/` sets `eval_metric="AUC"` — including the ones
this project intends to port (`tomasa2__*`, `donmarch14__s6e8-catboost.py`,
`abdullahsafwan333__s6e8-catboost-sap.py`). Copying them verbatim reintroduces the problem.

## Scope

- [ ] Drop `eval_metric="AUC"` from any `task_type="GPU"` CatBoost member; score AUC externally
      from the fold predictions instead
- [ ] Keep `eval_metric="AUC"` only where the member genuinely runs on CPU
- [ ] Set `thread_count` from `S6E8_THREADS` rather than leaving it at the default
- [ ] Keep `allow_writing_files=False` — the default litters `catboost_info/`
- [ ] If early stopping is wanted on GPU, drive it from `Logloss` and select the iteration by AUC
      computed outside the fit

## Watch for

CatBoost was also the heaviest member measured — **1,076 MB peak RSS** against 430–554 MB for the
other families. On a box with ~2 GB available that matters; do not run a CatBoost member
concurrently with anything else large.

## Next

Issue 005 (CatBoost native-categorical member) depends on this.
