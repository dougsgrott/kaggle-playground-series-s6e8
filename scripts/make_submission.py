"""Build a submission CSV from an exported member or stack artifact.

    uv run python scripts/make_submission.py --from xgb_baseline [--tag my_tag]

Reads oof/test_<name>.npy (the positional contract) and writes submissions/<tag>.csv.
Use --rank to submit percentile ranks instead of probabilities; ROC-AUC only reads
order, and ranks make a file directly blendable with another rank-space submission.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from s6e8 import config as C
from s6e8.data import load_train
from s6e8.submit import describe, write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="name", required=True, help="member/stack name in oof/")
    ap.add_argument("--tag", default=None, help="output name (defaults to --from)")
    ap.add_argument("--rank", action="store_true", help="write percentile ranks, not probabilities")
    args = ap.parse_args()

    test_path = C.OOF_DIR / f"test_{args.name}.npy"
    oof_path = C.OOF_DIR / f"oof_{args.name}.npy"
    if not test_path.exists():
        raise SystemExit(f"missing {test_path}")

    test_pred = np.load(test_path).astype(np.float64)
    if oof_path.exists():
        y = load_train()[C.TARGET].to_numpy(np.int8)
        oof = np.load(oof_path).astype(np.float64)
        print(f"pooled OOF AUC {roc_auc_score(y, oof):.6f}   "
              f"(nested honest CV runs ~0.0011 below LB; the offset is a line, not a "
              f"constant -- see docs/experiments.md)")

    if args.rank:
        test_pred = (rankdata(test_pred) - 0.5) / len(test_pred)

    path = write_submission(args.tag or args.name, test_pred)
    print(describe(path))


if __name__ == "__main__":
    main()
