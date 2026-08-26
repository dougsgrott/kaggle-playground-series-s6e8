"""Loading the competition data and the frozen fold assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s6e8 import config as C


def load_train() -> pd.DataFrame:
    df = pd.read_csv(C.TRAIN_CSV)
    assert len(df) == C.N_TRAIN, f"train has {len(df)} rows, expected {C.N_TRAIN}"
    return df


def load_test() -> pd.DataFrame:
    df = pd.read_csv(C.TEST_CSV)
    assert len(df) == C.N_TEST, f"test has {len(df)} rows, expected {C.N_TEST}"
    return df


def load_folds() -> np.ndarray:
    """Fold id per train row, in train.csv order. Written by scripts/make_folds.py."""
    folds = np.load(C.FOLDS_NPY)
    assert folds.shape == (C.N_TRAIN,), f"folds has shape {folds.shape}"
    return folds


def fold_splits(folds: np.ndarray | None = None):
    """Yield (train_idx, valid_idx) for each fold, matching the frozen contract."""
    if folds is None:
        folds = load_folds()
    all_idx = np.arange(len(folds))
    for k in range(C.N_SPLITS):
        yield all_idx[folds != k], all_idx[folds == k]


def read_decimal_places(path, cols) -> pd.DataFrame:
    """Number of printed decimal places per column, read from the RAW CSV TEXT.

    This has to happen before pandas parses the field as a float: float64 cannot tell
    you that "1.80" was written with two decimals and "1.8" with one. NaN where the
    field itself is missing.

    Ported from analysis/nb_clean/dariushafshar__s6e8-what-actually-helps.py
    """
    raw = pd.read_csv(path, usecols=list(cols), dtype={c: str for c in cols})
    out = pd.DataFrame(index=raw.index)
    for c in cols:
        s = raw[c]
        after = s.str.split(".", n=1).str[1]
        nd = pd.to_numeric(after.str.len(), errors="coerce").fillna(0.0)
        out[f"nd_{c}"] = nd.where(s.notna()).astype("float64")
    return out


def string_levels(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Exact-value string levels with missing as an EXPLICIT level.

    The obvious `df[c].astype(str)` is WRONG on pandas >= 3.0: the new str dtype
    preserves NA, so groupby silently drops every missing row from the level
    statistics and fillna(prior) quietly hands them the base rate. Routing through
    object and filling first gives missing its own level on both 2.x and 3.x.

    This environment runs pandas 3.x, so the trap is live. See docs/ROADMAP.md.
    """
    out = pd.DataFrame(
        {c: df[c].astype(object).fillna("__missing__").astype(str).values for c in cols}
    )
    assert len(out) == len(df)
    return out


def save_member(name: str, oof: np.ndarray, test: np.ndarray) -> None:
    """Write the positional OOF contract: no id column, file order, float32."""
    oof = np.asarray(oof, dtype=np.float32).reshape(-1)
    test = np.asarray(test, dtype=np.float32).reshape(-1)
    assert oof.shape == (C.N_TRAIN,), f"oof has shape {oof.shape}"
    assert test.shape == (C.N_TEST,), f"test has shape {test.shape}"
    assert np.isfinite(oof).all() and np.isfinite(test).all(), "non-finite predictions"
    C.OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.OOF_DIR / f"oof_{name}.npy", oof)
    np.save(C.OOF_DIR / f"test_{name}.npy", test)
