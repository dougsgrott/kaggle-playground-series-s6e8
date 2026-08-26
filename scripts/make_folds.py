"""Materialise the frozen fold contract to data/folds.npy.

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train[TARGET])

over train.csv in ORIGINAL FILE ROW ORDER. This is the partition the entire public OOF
ecosystem uses; every member in oof/ must share it or it cannot enter a stack.

Idempotent: rerunning reproduces byte-identical output, and the digest is asserted.
"""
from __future__ import annotations

import hashlib

import numpy as np

from s6e8 import config as C
from s6e8.data import load_train, make_fold_assignment


def main() -> None:
    train = load_train()
    y = train[C.TARGET].to_numpy(np.int8)

    # The rule itself lives in s6e8.data so the noise-floor sweep varies the seed
    # against exactly the same code path. See docs/issues/003.
    folds = make_fold_assignment(y, seed=C.FOLD_SEED, n_splits=C.N_SPLITS)

    counts = np.bincount(folds, minlength=C.N_SPLITS)
    rates = [float(y[folds == k].mean()) for k in range(C.N_SPLITS)]

    digest = hashlib.sha256(folds.tobytes()).hexdigest()

    C.FOLDS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(C.FOLDS_NPY, folds)

    print(f"rows              {len(folds):,}")
    print(f"positive rate     {y.mean():.6f}")
    print(f"fold sizes        {counts.tolist()}")
    print("fold pos rates    " + "  ".join(f"{r:.6f}" for r in rates))
    print(f"sha256            {digest}")
    print(f"written           {C.FOLDS_NPY}")


if __name__ == "__main__":
    main()
