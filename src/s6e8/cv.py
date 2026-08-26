"""Cross-validation harness over the frozen folds.

Every member goes through `run_member` so the positional OOF contract is enforced in
exactly one place: 691,369 predictions in train.csv order, 296,302 in test.csv order,
no id column. A member that cannot produce that cannot enter a stack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

from s6e8 import config as C
from s6e8.data import fold_splits, load_folds


class FitPredict(Protocol):
    """Fit on one fold and return (valid_pred, test_pred).

    Implementations receive positional indices into the full training frame, so they
    are free to slice whatever representation they hold.
    """

    def __call__(self, fold: int, train_idx: np.ndarray, valid_idx: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class MemberResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    y: np.ndarray
    fold_aucs: list[float] = field(default_factory=list)
    fold_seconds: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def oof_auc(self) -> float:
        """Pooled OOF AUC -- scored once over all rows, not averaged across folds."""
        return float(roc_auc_score(self.y, self.oof))

    def summary(self) -> str:
        folds = "  ".join(f"{a:.5f}" for a in self.fold_aucs)
        spread = max(self.fold_aucs) - min(self.fold_aucs)
        return (
            f"{self.name}\n"
            f"  pooled OOF AUC   {self.oof_auc:.6f}\n"
            f"  per-fold         {folds}\n"
            f"  fold mean        {np.mean(self.fold_aucs):.6f} "
            f"+/- {np.std(self.fold_aucs, ddof=1):.6f}\n"
            f"  fold range       {spread:.6f}   <-- NOT the noise floor; it overstates it\n"
            f"                                       by ~10x. See docs/issues/003.\n"
            f"  total time       {sum(self.fold_seconds) / 60:.1f} min"
        )


def run_member(name: str, fit_predict: FitPredict, y: np.ndarray,
               folds: np.ndarray | None = None, verbose: bool = True) -> MemberResult:
    """Run one member across the frozen folds and return the aligned OOF/test arrays.

    Test predictions are averaged over the fold models, which is why a member's OOF
    reads lower than its leaderboard score: OOF comes from one model per row, the
    submission from five. That gap is the CV->LB offset, and it shrinks as the model
    improves. See docs/experiments.md.
    """
    if folds is None:
        folds = load_folds()

    oof = np.zeros(C.N_TRAIN, dtype=np.float64)
    test = np.zeros(C.N_TEST, dtype=np.float64)
    result = MemberResult(name=name, oof=oof, test=test, y=y)

    for fold, (train_idx, valid_idx) in enumerate(fold_splits(folds)):
        t0 = time.perf_counter()
        valid_pred, test_pred = fit_predict(fold, train_idx, valid_idx)

        valid_pred = np.asarray(valid_pred, dtype=np.float64).reshape(-1)
        test_pred = np.asarray(test_pred, dtype=np.float64).reshape(-1)
        if valid_pred.shape != valid_idx.shape:
            raise ValueError(
                f"fold {fold}: got {valid_pred.shape[0]} valid predictions for "
                f"{valid_idx.shape[0]} rows")
        if test_pred.shape != (C.N_TEST,):
            raise ValueError(f"fold {fold}: test prediction has shape {test_pred.shape}")

        oof[valid_idx] = valid_pred
        test += test_pred / C.N_SPLITS

        auc = float(roc_auc_score(y[valid_idx], valid_pred))
        seconds = time.perf_counter() - t0
        result.fold_aucs.append(auc)
        result.fold_seconds.append(seconds)
        if verbose:
            print(f"  fold {fold}  AUC {auc:.6f}  {seconds:6.1f}s", flush=True)

    # Every row must have been written exactly once; a fold file that does not cover
    # the frame would otherwise leave zeros that still score plausibly.
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("non-finite predictions")
    return result
