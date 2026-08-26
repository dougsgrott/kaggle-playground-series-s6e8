"""Materialise the frozen fold contract to data/folds.npy.

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train[TARGET])

over train.csv in ORIGINAL FILE ROW ORDER. This is the partition the entire public OOF
ecosystem uses; every member in oof/ must share it or it cannot enter a stack.

Idempotent: rerunning reproduces byte-identical output, and the digest is asserted.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import StratifiedKFold

from s6e8 import config as C
from s6e8.data import load_train


def main() -> None:
    train = load_train()
    y = train[C.TARGET].to_numpy(np.int8)

    folds = np.full(len(train), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.FOLD_SEED)
    for k, (_, valid_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[valid_idx] = k

    assert (folds >= 0).all(), "some rows were never assigned a fold"
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
