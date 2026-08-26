"""Measure this machine's noise floor -- docs/issues/003.

    uv run python scripts/noise_floor.py [--out docs/noise_floor.json]

Answers two questions that both get called "the noise floor":

  sigma_partition   how far the CV number itself moves when the PARTITION changes,
                    model seed held fixed. Sweeps partition seeds. This is the
                    quantity the corpus reports as 0.00005.

  sigma_delta       how far an A-vs-B DELTA moves under the null, on the FROZEN
                    partition, with both arms re-seeded. Sweeps model seeds at
                    seed 42 and multiplies by sqrt(2). This is the null distribution
                    of every row in docs/experiments.md, since those hold the
                    partition fixed and vary only the config.

The per-fold range inside a single run is reported alongside both, because it is the
number people reach for by mistake and it is far larger than either.

data/folds.npy is never touched: alternate partitions are rebuilt in memory.

Cost: the model is the starter feature block at lr=0.10 x 900 fixed rounds, which
reaches ~0.9638 on fold 0 in ~31 s against the baseline member's 0.9642 in ~104 s.
Same AUC regime, a third of the price. Early stopping is removed on purpose -- it
picks its round on the same fold the AUC is read from, which is a separate source of
variance and would be attributed to the model seed here.
"""
from __future__ import annotations

import argparse

# Must precede any modelling import so the OpenMP pools are sized correctly.
from s6e8.runtime import configure_threads

THREADS = configure_threads()

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from s6e8 import config as C  # noqa: E402
from s6e8.cv import repeated_cv  # noqa: E402
from s6e8.data import load_test, load_train  # noqa: E402
from s6e8.models import STARTER_XGB_PARAMS, starter_matrices  # noqa: E402
from s6e8.runtime import assert_xgboost_gpu, n_threads  # noqa: E402

# Fixed-round stand-in for the baseline member. Same shrinkage budget (0.10 x 900 =
# 90 vs 0.035 x ~2700 = 95), no early stopping.
PROBE_PARAMS = {k: v for k, v in STARTER_XGB_PARAMS.items()
                if k not in ("n_estimators", "learning_rate", "early_stopping_rounds",
                             "random_state")}
PROBE_PARAMS.update(learning_rate=0.10, n_estimators=900)

PARTITION_SEEDS = (42, 2024, 7, 1337, 31337, 5150)
MODEL_SEEDS = (42, 2024, 7, 1337, 31337, 5150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "docs" / "noise_floor.json"))
    ap.add_argument("--partition-seeds", type=int, nargs="+", default=list(PARTITION_SEEDS))
    ap.add_argument("--model-seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    ap.add_argument("--rounds", type=int, default=None,
                    help="override n_estimators; only for smoke-testing the harness")
    args = ap.parse_args()
    if args.rounds:
        PROBE_PARAMS["n_estimators"] = args.rounds

    t0 = time.perf_counter()
    print(f"noise floor   threads {THREADS}", flush=True)
    print(f"  {assert_xgboost_gpu()}", flush=True)

    import xgboost as xgb

    train, test = load_train(), load_test()
    y = train[C.TARGET].to_numpy(np.int8)
    X, _ = starter_matrices(train, test)
    del test
    print(f"  design matrix {X.shape}  "
          f"{X.memory_usage(deep=True).sum() / 1024 ** 2:.0f} MB", flush=True)
    print(f"  model lr={PROBE_PARAMS['learning_rate']} "
          f"n_estimators={PROBE_PARAMS['n_estimators']} (no early stopping)", flush=True)

    def build(model_seed: int):
        params = dict(PROBE_PARAMS, device="cuda", n_jobs=n_threads(),
                      random_state=model_seed)

        def fit_score(fold, train_idx, valid_idx):
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
            return model.predict_proba(X.iloc[valid_idx])[:, 1]

        return fit_score

    # A cross, not a grid: partitions at the reference model seed, then model seeds at
    # the frozen partition. The shared (42, 42) cell is run once.
    ref_model, frozen = args.model_seeds[0], C.FOLD_SEED
    cells = [(p, ref_model) for p in args.partition_seeds]
    cells += [(frozen, m) for m in args.model_seeds
              if (frozen, m) not in cells]
    print(f"  {len(cells)} runs x {C.N_SPLITS} folds = {len(cells) * C.N_SPLITS} fits\n",
          flush=True)

    res = repeated_cv(build, y, cells)

    sig_p_pooled, sig_p_mean = res.partition_sigma(ref_model)
    sig_m = res.model_sigma(frozen)
    sig_d = res.delta_sigma(frozen)
    max_null = res.max_null_delta(frozen)
    fold_range = res.mean_fold_range()
    p_cells = res.select(model_seed=ref_model)
    m_cells = res.select(partition_seed=frozen)

    print(f"""
================================================================================
NOISE FLOOR   ({len(p_cells)} partition seeds, {len(m_cells)} model seeds, {len(cells) * C.N_SPLITS} fits)
================================================================================

sigma_partition   {sig_p_pooled:.6f}   pooled AUC across partition seeds (model seed {ref_model})
                  {sig_p_mean:.6f}   ...of the fold MEANS, which is what issue 003 asked for
   pooled AUCs    {"  ".join(f"{c.pooled_auc:.6f}" for c in p_cells)}
   seeds          {"    ".join(f"{c.partition_seed:<8d}" for c in p_cells)}

sigma_model       {sig_m:.6f}   pooled AUC across model seeds at the frozen partition
sigma_delta       {sig_d:.6f}   = sqrt(2) * sigma_model -- the null spread of an A/B delta
max null delta    {max_null:.6f}   largest gap actually produced by changing NOTHING
   pooled AUCs    {"  ".join(f"{c.pooled_auc:.6f}" for c in m_cells)}
   seeds          {"    ".join(f"{c.model_seed:<8d}" for c in m_cells)}

mean fold range   {fold_range:.6f}   <-- NOT a noise floor. {fold_range / sig_d:.0f}x sigma_delta.

USE
   ablation on the frozen folds        must clear  {2 * sig_d:.6f}  (2 sigma_delta)
   comparison across partitions/CV     must clear  {2 * sig_p_pooled:.6f}  (2 sigma_partition)
================================================================================""",
          flush=True)

    payload = {
        "measured": time.strftime("%Y-%m-%d"),
        "model": {"family": "xgboost", "device": "cuda", **PROBE_PARAMS,
                  "features": "starter block, 34 columns"},
        "n_fits": len(cells) * C.N_SPLITS,
        "sigma_partition_pooled": sig_p_pooled,
        "sigma_partition_foldmean": sig_p_mean,
        "sigma_model": sig_m,
        "sigma_delta": sig_d,
        "max_null_delta": max_null,
        "mean_fold_range": fold_range,
        "cells": [{"partition_seed": c.partition_seed, "model_seed": c.model_seed,
                   "pooled_auc": c.pooled_auc, "fold_mean": c.fold_mean,
                   "fold_aucs": c.fold_aucs, "fold_range": c.fold_range,
                   "seconds": c.seconds} for c in res.cells],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten  {args.out}", flush=True)
    print(f"total    {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
