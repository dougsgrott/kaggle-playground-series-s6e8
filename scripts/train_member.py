"""Train one registered member on the frozen folds and export the OOF contract.

    uv run python scripts/train_member.py --model xgb_baseline [--submit-tag xgb_baseline]

Writes oof/oof_<name>.npy and oof/test_<name>.npy, and optionally a submission CSV.
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
from s6e8.cv import run_member  # noqa: E402
from s6e8.data import load_folds, load_test, load_train, save_member  # noqa: E402
from s6e8.models import REGISTRY  # noqa: E402
from s6e8.submit import describe, write_submission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--submit-tag", default=None,
                    help="also write submissions/<tag>.csv from the fold-averaged test predictions")
    args = ap.parse_args()

    print(f"model {args.model}   threads {THREADS}", flush=True)
    t0 = time.perf_counter()

    train, test = load_train(), load_test()
    y = train[C.TARGET].to_numpy(np.int8)
    folds = load_folds()

    fit_predict = REGISTRY[args.model](train, test, y)
    result = run_member(args.model, fit_predict, y, folds)

    save_member(args.model, result.oof, result.test)
    print("\n" + result.summary(), flush=True)

    extra = dict(getattr(fit_predict, "extra", {}))
    if extra:
        print(f"  {json.dumps(extra)}", flush=True)

    if args.submit_tag:
        path = write_submission(args.submit_tag, result.test)
        print("\n" + describe(path), flush=True)

    print(f"\ntotal {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
