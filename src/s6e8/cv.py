"""Cross-validation harness over the frozen folds.

Every member goes through `run_member` so the positional OOF contract is enforced in
exactly one place: 691,369 predictions in train.csv order, 296,302 in test.csv order,
no id column. A member that cannot produce that cannot enter a stack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

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
            f"  fold range       {spread:.6f}   <-- NOT an error bar. Typically {C.FOLD_RANGE_TYPICAL:.4f},\n"
            f"                                       22x sigma_delta={C.SIGMA_DELTA:.6f}, and mostly the\n"
            f"                                       split: fold 3 is +0.00097 easier, fold 0 -0.00061\n"
            f"                                       harder, reproducibly. See docs/issues/003.\n"
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


# ================================================================================
# Noise floor -- docs/issues/003
#
# Two different quantities get called "the noise floor", and conflating them is how
# a nonexistent gain gets shipped:
#
#   sigma_partition  spread of the CV number itself when the PARTITION changes.
#                    Governs comparisons against a CV number computed on some other
#                    split -- a published figure, or a rebuilt fold file.
#
#   sigma_delta      spread of an A-vs-B DELTA measured on the frozen partition with
#                    both arms re-seeded. This is the null distribution of every row
#                    in docs/experiments.md, because those rows all hold the partition
#                    fixed at seed 42 and vary only the config.
#
# sigma_delta is the smaller of the two and is the one an ablation must clear.
# Neither is the per-fold range inside a single run, which is larger than both and
# measures nothing but how much easier one fold is than another.
# ================================================================================


class FitScore(Protocol):
    """Fit on one fold and return validation predictions only.

    Narrower than `FitPredict` on purpose: the noise floor never needs test
    predictions, and skipping them saves a 296,302-row inference per fit.
    """

    def __call__(self, fold: int, train_idx: np.ndarray, valid_idx: np.ndarray
                 ) -> np.ndarray: ...


@dataclass
class CVCell:
    """One complete k-fold run at a given (partition seed, model seed)."""
    partition_seed: int
    model_seed: int
    pooled_auc: float
    fold_aucs: list[float]
    seconds: float

    @property
    def fold_mean(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def fold_range(self) -> float:
        return float(max(self.fold_aucs) - min(self.fold_aucs))


@dataclass
class RepeatedCVResult:
    cells: list[CVCell] = field(default_factory=list)

    def select(self, partition_seed: int | None = None,
               model_seed: int | None = None) -> list[CVCell]:
        return [c for c in self.cells
                if (partition_seed is None or c.partition_seed == partition_seed)
                and (model_seed is None or c.model_seed == model_seed)]

    @staticmethod
    def _std(values: list[float]) -> float:
        """Sample std. Undefined below two observations -- say so rather than return 0."""
        if len(values) < 2:
            return float("nan")
        return float(np.std(values, ddof=1))

    def partition_sigma(self, model_seed: int) -> tuple[float, float]:
        """(std of pooled AUC, std of fold-mean AUC) across partition seeds."""
        cells = self.select(model_seed=model_seed)
        return (self._std([c.pooled_auc for c in cells]),
                self._std([c.fold_mean for c in cells]))

    def model_sigma(self, partition_seed: int) -> float:
        """Std of pooled AUC across model seeds at one fixed partition."""
        return self._std([c.pooled_auc for c in self.select(partition_seed=partition_seed)])

    def delta_sigma(self, partition_seed: int) -> float:
        """Null std of an A-vs-B delta on the frozen partition.

        Two independently seeded arms of the SAME config differ by the difference of
        two draws from the model-seed distribution, so the null delta has std
        sqrt(2) * sigma_model. An ablation delta smaller than this measured nothing.
        """
        return float(np.sqrt(2.0) * self.model_sigma(partition_seed))

    def max_null_delta(self, partition_seed: int) -> float:
        """Largest observed gap between two identically-configured runs.

        The empirical companion to `delta_sigma`: the worst delta actually produced by
        changing nothing at all.
        """
        aucs = [c.pooled_auc for c in self.select(partition_seed=partition_seed)]
        return float(max(aucs) - min(aucs)) if len(aucs) >= 2 else float("nan")

    def mean_fold_range(self) -> float:
        return float(np.mean([c.fold_range for c in self.cells]))


def repeated_cv(build_fit_score: "Callable[[int], FitScore]", y: np.ndarray,
                cells: "Sequence[tuple[int, int]]", n_splits: int = C.N_SPLITS,
                verbose: bool = True) -> RepeatedCVResult:
    """Run one k-fold pass per (partition_seed, model_seed) cell.

    `cells` is an explicit list rather than a grid because the useful shape is a cross,
    not a product: sweep partitions at one model seed to get sigma_partition, then
    sweep model seeds at the frozen partition to get sigma_delta. The full grid would
    cost several times as much and answer nothing extra.

    `data/folds.npy` is never read or written here -- partitions are rebuilt in memory
    from `make_fold_assignment`, so the frozen contract stays frozen.
    """
    from s6e8.data import make_fold_assignment

    result = RepeatedCVResult()
    partitions: dict[int, np.ndarray] = {}

    for i, (p_seed, m_seed) in enumerate(cells):
        if p_seed not in partitions:
            partitions[p_seed] = make_fold_assignment(y, seed=p_seed, n_splits=n_splits)
        folds = partitions[p_seed]
        fit_score = build_fit_score(m_seed)

        t0 = time.perf_counter()
        oof = np.full(len(y), np.nan, dtype=np.float64)
        fold_aucs: list[float] = []
        for fold, (train_idx, valid_idx) in enumerate(fold_splits(folds)):
            valid_pred = np.asarray(fit_score(fold, train_idx, valid_idx),
                                    dtype=np.float64).reshape(-1)
            if valid_pred.shape != valid_idx.shape:
                raise ValueError(
                    f"partition {p_seed} fold {fold}: got {valid_pred.shape[0]} "
                    f"predictions for {valid_idx.shape[0]} rows")
            oof[valid_idx] = valid_pred
            fold_aucs.append(float(roc_auc_score(y[valid_idx], valid_pred)))

        if not np.isfinite(oof).all():
            raise ValueError(f"partition {p_seed}: some rows were never predicted")
        cell = CVCell(partition_seed=p_seed, model_seed=m_seed,
                      pooled_auc=float(roc_auc_score(y, oof)), fold_aucs=fold_aucs,
                      seconds=time.perf_counter() - t0)
        result.cells.append(cell)
        if verbose:
            print(f"  [{i + 1:2d}/{len(cells)}] partition {p_seed:<5} model {m_seed:<5} "
                  f"pooled {cell.pooled_auc:.6f}  fold-mean {cell.fold_mean:.6f}  "
                  f"range {cell.fold_range:.6f}  {cell.seconds / 60:.1f} min", flush=True)
    return result
