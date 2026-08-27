"""Rank-correlation matrix and admission verdicts over everything in oof/ — issue 007.

    uv run python scripts/member_matrix.py [--members a b c]

The admission rule (roadmap, measured in the corpus): a member must be **decorrelated
AND comparably strong**. Rank correlation above ~0.99 to an existing member, or more than
~0.006 AUC weaker than the best, and it earns weight 0.000. There is a visible strength
cliff near solo OOF 0.966 — blend contribution tracks solo OOF more than it tracks
decorrelation, so "different but weak" is the worse failure of the two.

Correlation is Spearman on the OOF vector, because the metric is ROC-AUC and AUC depends
only on ranks. Pearson on probabilities would report agreement the metric cannot see.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from s6e8 import config as C
from s6e8.data import load_train

CORR_CEILING = 0.99      # above this, a member duplicates one already present
STRENGTH_FLOOR = 0.006   # more than this below the best, and it earns weight 0.000


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. Computed once per member via rankdata, then Pearson on ranks."""
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--blend-check", action="store_true",
                    help="test whether the corr rejection predicts zero stack weight")
    args = ap.parse_args()

    names = args.members or sorted(
        p.name[len("oof_"):-len(".npy")] for p in C.OOF_DIR.glob("oof_*.npy"))
    y = load_train()[C.TARGET].to_numpy(np.int8)

    oof = {n: np.load(C.OOF_DIR / f"oof_{n}.npy").astype(np.float64) for n in names}
    for n, v in oof.items():
        assert v.shape == (C.N_TRAIN,), f"{n}: shape {v.shape} breaks the OOF contract"
        assert np.isfinite(v).all(), f"{n}: non-finite OOF"
    aucs = {n: float(roc_auc_score(y, v)) for n, v in oof.items()}
    ranks = {n: rankdata(v) for n, v in oof.items()}

    best = max(aucs.values())
    order = sorted(names, key=lambda n: -aucs[n])

    print(f"{len(names)} members on the frozen folds, {C.N_TRAIN:,} rows each\n")
    print(f"{'member':16s} {'solo OOF':>10s} {'vs best':>9s} {'max corr':>9s} "
          f"{'to':16s}  admission")
    print("-" * 78)

    verdicts = {}
    for i, n in enumerate(order):
        # Compare only against members already admitted ahead of it: a member is
        # redundant relative to what the stack HAS, not to everything that exists.
        earlier = [m for m in order[:i] if verdicts.get(m, "").startswith("ADMIT")]
        if earlier:
            corrs = {m: spearman(ranks[n], ranks[m]) for m in earlier}
            top, cmax = max(corrs.items(), key=lambda kv: kv[1])
        else:
            top, cmax = "-", float("nan")

        gap = aucs[n] - best
        if not earlier:
            v = "ADMIT (anchor)"
        elif cmax > CORR_CEILING:
            v = f"REJECT corr>{CORR_CEILING}"
        elif gap < -STRENGTH_FLOOR:
            v = f"REJECT weak>{STRENGTH_FLOOR}"
        else:
            v = "ADMIT"
        verdicts[n] = v
        cm = "  n/a  " if np.isnan(cmax) else f"{cmax:9.4f}"
        print(f"{n:16s} {aucs[n]:10.6f} {gap:+9.6f} {cm} {top:16s}  {v}")

    print("\nrank-correlation matrix")
    w = max(len(n) for n in order) + 1
    print(" " * w + "".join(f"{n[:9]:>10s}" for n in order))
    for a in order:
        row = "".join(f"{spearman(ranks[a], ranks[b]):10.4f}" for b in order)
        print(f"{a:{w}s}{row}")

    pairs = [(a, b, spearman(ranks[a], ranks[b])) for a, b in itertools.combinations(order, 2)]
    if pairs:
        lo = min(pairs, key=lambda t: t[2])
        print(f"\nmost decorrelated pair: {lo[0]} / {lo[1]}  rho {lo[2]:.4f}")
        print(f"admitted: {sum(v.startswith('ADMIT') for v in verdicts.values())} "
              f"of {len(order)}")

    if args.blend_check and len(order) > 1:
        from s6e8.data import load_folds
        blend_check(order, oof, y, load_folds(), order[0])


def blend_check(names, oof, y, folds, anchor: str) -> None:
    """Does the rho>0.99 rejection actually predict zero stack weight HERE?

    The admission rule is a corpus heuristic. This turns it into a measurement: fit an
    L2 logistic blend in logit space under NESTED CV -- weights fitted on 4 folds, scored
    on the 5th -- and report what each rejected member adds on top of the anchor. Naive
    full-OOF weight fitting reports a premium that does not exist out of sample
    (non-negotiable #3), so it is not used even for a quick check.
    """
    from sklearn.linear_model import LogisticRegression

    def logit(v):
        v = np.clip(v, 1e-6, 1 - 1e-6)
        return np.log(v / (1 - v))

    L = {n: logit(oof[n]) for n in names}

    def nested(members) -> float:
        pred = np.zeros(len(y))
        X = np.column_stack([L[m] for m in members])
        for k in range(C.N_SPLITS):
            tr, va = folds != k, folds == k
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(X[tr], y[tr])
            pred[va] = lr.decision_function(X[va])
        return float(roc_auc_score(y, pred))

    base = nested([anchor])
    print(f"\nnested-CV blend check (weights fitted on 4 folds, scored on the 5th)")
    print(f"  {'members':44s} {'nested OOF':>11s} {'vs anchor':>10s}")
    print(f"  {anchor + ' (anchor alone)':44s} {base:11.6f} {'':>10s}")
    others = [n for n in names if n != anchor]
    for m in others:
        a = nested([anchor, m])
        print(f"  {anchor + ' + ' + m:44s} {a:11.6f} {a - base:+10.6f}")
    allm = nested([anchor] + others)
    print(f"  {'all ' + str(len(names)) + ' members':44s} {allm:11.6f} {allm - base:+10.6f}")


if __name__ == "__main__":
    main()
