"""Measure real per-member training cost and peak memory so Phase 2/3 planning rests on data.

Times one fold of each GBDT family on the raw 12 columns and reports RSS at every step.
Deliberately memory-lean: this machine has ~7 GB nominal but frequently under 3 GB free,
and the fusion layer is the part that will actually run out. Each family is guarded so one
failure does not lose the whole run.

Not a model search. 400 rounds is a timing probe; real members run 3000-8000.
"""
from __future__ import annotations

import os

# Must be set BEFORE lightgbm/xgboost import: their OpenMP pools size themselves at load.
# This box has 20 cores but runs at load average ~50, so extra threads spin and contend --
# measured 2.0s at 1 thread vs 18.0s at 4 for the same XGBoost fit. See docs/issues/001.
THREADS = os.environ.setdefault("S6E8_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)

import gc
import resource
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8 import config as C
from s6e8.data import fold_splits, load_folds, load_train

ROUNDS = 400
ROWS = []


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def avail_mb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def stamp(label: str) -> None:
    print(f"    [{label:<22}] rss {rss_mb():7.0f} MB   avail {avail_mb():7.0f} MB", flush=True)


def record(name, seconds, auc, note=""):
    ROWS.append({
        "member": name,
        "1 fold (s)": round(seconds, 1),
        "5 folds (min)": round(5 * seconds / 60, 1),
        "fold AUC": round(auc, 5),
        "peak RSS (MB)": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "note": note,
    })
    print(f"  -> {name:<28} {seconds:7.1f}s   AUC {auc:.5f}   {note}", flush=True)


def main() -> None:
    print(f"cores {os.cpu_count()}   threads {THREADS}   loadavg {os.getloadavg()[0]:.1f}"
          f"   avail {avail_mb():.0f} MB at start\n", flush=True)

    train = load_train()
    y = train[C.TARGET].to_numpy(np.int8)
    X = train[C.FEATURES].copy()
    for c in C.NUM_COLS:
        X[c] = X[c].astype(np.float32)
    for c in C.CAT_COLS:
        X[c] = X[c].astype("category")
    del train
    gc.collect()

    tr, va = next(iter(fold_splits(load_folds())))
    print(f"one fold: {len(tr):,} train / {len(va):,} valid, {X.shape[1]} raw features")
    stamp("data loaded")
    print()

    # --- XGBoost, GPU, native categorical (no one-hot copy) -------------------
    try:
        import xgboost as xgb
        cuda = xgb.build_info().get("CUDA_VERSION")
        print(f"xgboost {xgb.__version__} (built for CUDA {cuda}) GPU", flush=True)
        # 3.x wheels are built for CUDA 13 and fall back to CPU on this 12.9 driver,
        # with a warning rather than an error. Fail loudly instead.
        assert cuda and cuda[0] == 12, f"xgboost built for CUDA {cuda}; pin xgboost<3.0"
        t0 = time.perf_counter()
        m = xgb.XGBClassifier(
            n_estimators=ROUNDS, learning_rate=0.05, max_depth=7, min_child_weight=15,
            subsample=0.85, colsample_bytree=0.85, tree_method="hist", device="cuda",
            enable_categorical=True, max_cat_to_onehot=8, eval_metric="auc",
            random_state=C.FOLD_SEED, verbosity=0,
        )
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        record("xgboost GPU", time.perf_counter() - t0, roc_auc_score(y[va], p))
        del m, p
        gc.collect()
        stamp("after xgboost")
    except Exception as exc:  # noqa: BLE001 - a benchmark must survive one family failing
        print(f"  !! xgboost failed: {type(exc).__name__}: {exc}", flush=True)

    # --- LightGBM, CPU, at both max_bin settings ------------------------------
    for max_bin in (255, 2047):
        try:
            import lightgbm as lgb
            print(f"\nlightgbm CPU max_bin={max_bin}", flush=True)
            t0 = time.perf_counter()
            m = lgb.LGBMClassifier(
                n_estimators=ROUNDS, learning_rate=0.05, num_leaves=255,
                min_child_samples=100, colsample_bytree=0.8, subsample=0.8,
                subsample_freq=1, max_bin=max_bin, random_state=C.FOLD_SEED,
                n_jobs=int(THREADS), verbose=-1, force_row_wise=True,
            )
            m.fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            record(f"lightgbm max_bin={max_bin}", time.perf_counter() - t0,
                   roc_auc_score(y[va], p), "the max_bin lever" if max_bin == 2047 else "")
            del m, p
            gc.collect()
            stamp(f"after lgb {max_bin}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! lightgbm max_bin={max_bin} failed: {type(exc).__name__}: {exc}", flush=True)

    # --- CatBoost, GPU, native string levels ----------------------------------
    try:
        from catboost import CatBoostClassifier
        print("\ncatboost GPU (native string levels)", flush=True)
        Xc = X.copy()
        for c in C.CAT_COLS:
            Xc[c] = Xc[c].astype(object).fillna("__missing__").astype(str)
        cat_idx = [Xc.columns.get_loc(c) for c in C.CAT_COLS]
        t0 = time.perf_counter()
        m = CatBoostClassifier(
            iterations=ROUNDS, learning_rate=0.05, depth=6,
            random_seed=C.FOLD_SEED, verbose=0, task_type="GPU", thread_count=int(THREADS),
            allow_writing_files=False,
        )
        m.fit(Xc.iloc[tr], y[tr], cat_features=cat_idx, verbose=0)
        p = m.predict_proba(Xc.iloc[va])[:, 1]
        record("catboost GPU", time.perf_counter() - t0, roc_auc_score(y[va], p))
        del m, p, Xc
        gc.collect()
        stamp("after catboost")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! catboost failed: {type(exc).__name__}: {exc}", flush=True)

    print("\n" + (pd.DataFrame(ROWS).to_string(index=False) if ROWS else "no members completed"))
    print(f"\npeak RSS for the whole run: "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    print(f"{ROUNDS} rounds is a timing probe, not a converged model. Real members run "
          f"3000-8000; scale accordingly.")


if __name__ == "__main__":
    main()
