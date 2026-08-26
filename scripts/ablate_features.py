"""Price each feature block against the measured noise floor — docs/issues/002.

    uv run python scripts/ablate_features.py [--seeds 1] [--arms raw budget ...]

Cumulative ablation: each arm adds one block to the one before it, so every delta
answers the decision that actually matters — *does adding this block on top of what we
already have pay?* — rather than the block's value in isolation.

The gate is **2 * sigma_delta = 0.00011** (issue 003). Three verdicts, not two:

    PAYS        delta >= +2 sigma
    UNRESOLVED  |delta| < 2 sigma   -- measured nothing; NOT a win and NOT a loss
    HURTS       delta <= -2 sigma

The model is the same fixed-round XGBoost the noise floor was measured on (lr 0.10 x 900,
no early stopping), so the gate transfers. The one difference is `enable_categorical`,
since these blocks keep native categoricals instead of one-hotting. The baseline arm is
therefore run with several model seeds to re-measure sigma_model on this exact pipeline
and confirm the borrowed gate still applies.
"""
from __future__ import annotations

import argparse

# Must precede any modelling import so the OpenMP pools are sized correctly.
from s6e8.runtime import configure_threads

THREADS = configure_threads()

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from s6e8 import config as C  # noqa: E402
from s6e8.data import load_folds, load_test, load_train  # noqa: E402
from s6e8.features import NestedTargetEncoder, build_static  # noqa: E402
from s6e8.runtime import assert_xgboost_gpu, n_threads  # noqa: E402

# Identical to scripts/noise_floor.py so the measured gate applies unchanged.
FIT_PARAMS = dict(
    n_estimators=900, learning_rate=0.10, max_depth=7, min_child_weight=15,
    subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=3.0,
    objective="binary:logistic", eval_metric="auc", tree_method="hist",
    enable_categorical=True, verbosity=0,
)

# (label, cumulative block list, use target encoding)
_CHAIN = ["raw", "budget", "cat_isna", "lattice", "decimals", "freq", "impute"]
ARMS = [(name, tuple(_CHAIN[:i + 1]), False) for i, name in enumerate(_CHAIN)]
# The target encoder adds no static columns; it rides on the full static set.
ARMS.append(("te", tuple(_CHAIN), True))
MODEL_SEEDS = (42, 2024, 7, 1337, 31337)


def run_arm(X, y, folds, encoder, use_te, seed):
    """One 5-fold pass. Returns (pooled AUC, per-fold AUCs, seconds)."""
    import xgboost as xgb

    t0 = time.perf_counter()
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_aucs = []
    idx = np.arange(len(y))
    for k in range(C.N_SPLITS):
        tr_i, va_i = idx[folds != k], idx[folds == k]
        Xa, Xb = X.iloc[tr_i].reset_index(drop=True), X.iloc[va_i].reset_index(drop=True)
        if use_te:
            # Rebuilt inside the fold; the encoder never sees validation labels.
            e_tr, e_va, _ = encoder.build(tr_i, va_i)
            Xa = pd.concat([Xa, e_tr], axis=1)
            Xb = pd.concat([Xb, e_va], axis=1)
        model = xgb.XGBClassifier(**FIT_PARAMS, device="cuda", n_jobs=n_threads(),
                                  random_state=seed)
        model.fit(Xa, y[tr_i], verbose=False)
        pred = model.predict_proba(Xb)[:, 1]
        oof[va_i] = pred
        fold_aucs.append(float(roc_auc_score(y[va_i], pred)))
        del Xa, Xb
    assert np.isfinite(oof).all(), "some rows were never predicted"
    return float(roc_auc_score(y, oof)), fold_aucs, time.perf_counter() - t0


def verdict(delta: float, gate: float) -> str:
    if delta >= gate:
        return "PAYS"
    if delta <= -gate:
        return "HURTS"
    return "UNRESOLVED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1,
                    help="model seeds per arm; the baseline arm always gets >= 3")
    ap.add_argument("--multi", nargs="+", default=(),
                    help="arms to run with --multi-seeds instead of --seeds; spend this "
                         "on the blocks whose delta lands near the gate")
    ap.add_argument("--multi-seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=None, help="subset of arm labels")
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "docs" / "ablation_002.json"))
    ap.add_argument("--rounds", type=int, default=None,
                    help="override n_estimators; only for smoke-testing the harness")
    args = ap.parse_args()
    if args.rounds:
        FIT_PARAMS["n_estimators"] = args.rounds

    print(f"feature ablation   threads {THREADS}", flush=True)
    print(f"  {assert_xgboost_gpu()}", flush=True)
    t0 = time.perf_counter()

    train, test = load_train(), load_test()
    y = train[C.TARGET].to_numpy(np.int8)
    folds = load_folds()
    fs = build_static(train, test, n_jobs=n_threads())
    encoder = NestedTargetEncoder(train, test, y)
    del test

    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    gate = 2 * C.SIGMA_DELTA
    print(f"\n  gate 2*sigma_delta = {gate:.6f}\n", flush=True)

    results = []
    for label, blocks, use_te in arms:
        X = fs.train[fs.columns(blocks)]
        n_seeds = args.multi_seeds if label in args.multi else args.seeds
        if label == "raw":
            n_seeds = max(n_seeds, 3)
        seeds = MODEL_SEEDS[:n_seeds]
        aucs = []
        for s in seeds:
            auc, fold_aucs, secs = run_arm(X, y, folds, encoder, use_te, s)
            aucs.append(auc)
            n_cols = X.shape[1] + (len(encoder.names) if use_te else 0)
            print(f"  {label:9s} seed {s:<6} cols {n_cols:3d}  pooled {auc:.6f}  "
                  f"range {max(fold_aucs)-min(fold_aucs):.6f}  {secs/60:.1f} min",
                  flush=True)
        results.append({"label": label, "blocks": list(blocks), "use_te": use_te,
                        "n_cols": X.shape[1] + (len(encoder.names) if use_te else 0),
                        "seeds": list(seeds), "aucs": aucs,
                        "mean_auc": float(np.mean(aucs))})
        del X
        with open(args.out, "w") as f:
            json.dump({"gate": gate, "sigma_delta": C.SIGMA_DELTA,
                       "fit_params": FIT_PARAMS, "arms": results}, f, indent=2)

    print(f"\n{'=' * 88}\nCUMULATIVE ABLATION   gate 2*sigma_delta = {gate:.6f}\n{'=' * 88}")
    print(f"{'arm':10s} {'cols':>4s} {'seeds':>5s} {'pooled AUC':>11s} "
          f"{'Δ vs prev':>10s} {'× σ_d':>7s}  verdict")
    prev = None
    for r in results:
        n = len(r["seeds"])
        # Averaging n seeds shrinks each arm's own noise by sqrt(n); a delta between two
        # n-seed arms therefore has null sd sigma_delta / sqrt(n).
        g = gate / np.sqrt(n)
        d = "" if prev is None else f"{r['mean_auc'] - prev:+.6f}"
        x = "" if prev is None else f"{(r['mean_auc'] - prev) / C.SIGMA_DELTA:+.1f}"
        v = "baseline" if prev is None else verdict(r["mean_auc"] - prev, g)
        print(f"{r['label']:10s} {r['n_cols']:4d} {n:5d} {r['mean_auc']:11.6f} "
              f"{d:>10s} {x:>7s}  {v}")
        prev = r["mean_auc"]

    # sigma_model is NOT a constant -- it grows with the feature set, so report it
    # wherever it was measured rather than borrowing one number across the table.
    measured = [r for r in results if len(r["aucs"]) >= 2]
    if measured:
        print(f"\nsigma_model by feature set (noise-floor run, 34 one-hot cols: "
              f"{C.SIGMA_MODEL:.6f})")
        print(f"  {'arm':10s} {'cols':>4s} {'n':>2s} {'sigma_model':>12s} "
              f"{'sigma_delta':>12s} {'gate 2sd':>10s}")
        for r in measured:
            sm = float(np.std(r["aucs"], ddof=1))
            print(f"  {r['label']:10s} {r['n_cols']:4d} {len(r['aucs']):2d} "
                  f"{sm:12.6f} {np.sqrt(2)*sm:12.6f} {2*np.sqrt(2)*sm:10.6f}")
    if len(results) > 1:
        tot = results[-1]["mean_auc"] - results[0]["mean_auc"]
        print(f"\ntotal vs raw: {tot:+.6f}  ({tot / C.SIGMA_DELTA:+.0f} sigma_delta)")
    print(f"\nwritten  {args.out}")
    print(f"total    {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
