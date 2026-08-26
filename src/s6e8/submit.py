"""Turning a member or stack artifact into a submission CSV.

Every assertion here has to pass before anything is uploaded. Submissions are capped
at 10/day, so a malformed file costs a slot that cannot be recovered.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from s6e8 import config as C


def write_submission(tag: str, test_pred: np.ndarray, out_dir: Path | None = None) -> Path:
    """Write submissions/<tag>.csv, aligned to sample_submission's id order."""
    test_pred = np.asarray(test_pred, dtype=np.float64).reshape(-1)
    if test_pred.shape != (C.N_TEST,):
        raise ValueError(f"expected {C.N_TEST} predictions, got {test_pred.shape[0]}")
    if not np.isfinite(test_pred).all():
        raise ValueError("non-finite predictions")

    sample = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)
    test_ids = pd.read_csv(C.TEST_CSV, usecols=[C.ID_COL])[C.ID_COL].to_numpy()
    # The contract stores test predictions in test.csv row order; the submission must
    # be in sample_submission order. They agree today, but assert rather than assume.
    if not np.array_equal(sample[C.ID_COL].to_numpy(), test_ids):
        raise ValueError("sample_submission id order differs from test.csv row order")

    sub = pd.DataFrame({C.ID_COL: test_ids, C.TARGET: test_pred})
    if not sub[C.ID_COL].is_unique:
        raise ValueError("duplicate ids")
    if len(sub) != C.N_TEST:
        raise ValueError(f"submission has {len(sub)} rows")

    out_dir = out_dir or C.SUBMISSION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.csv"
    sub.to_csv(path, index=False)
    return path


def describe(path: Path) -> str:
    sub = pd.read_csv(path)
    v = sub[C.TARGET].to_numpy()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return (f"{path}  rows {len(sub):,}\n"
            f"  mean {v.mean():.5f}  (train positive rate {0.709424:.5f})\n"
            f"  min {v.min():.6f}   max {v.max():.6f}   distinct {len(np.unique(v)):,}\n"
            f"  sha256 {digest}")
