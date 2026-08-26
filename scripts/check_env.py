"""Phase 0 gate: libraries, GPU, data shapes, and the invariants the roadmap relies on.

Run this after any environment change. Every assertion here corresponds to a documented
fact in docs/data-notes.md; a failure means a doc is wrong or the environment is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s6e8 import config as C

PROJECT_VENV = C.PROJECT_ROOT / ".venv/lib/python3.12/site-packages"
from s6e8.data import load_folds, load_test, load_train, string_levels

OK, FAIL = "  ok  ", " FAIL "


def check(label, condition, detail=""):
    print(f"[{OK if condition else FAIL}] {label:<52} {detail}")
    return bool(condition)


def main() -> None:
    results = []

    print("\n== libraries ==")
    import catboost, lightgbm, torch, xgboost

    print(f"         pandas {pd.__version__}   numpy {np.__version__}")
    print(f"         lightgbm {lightgbm.__version__}   xgboost {xgboost.__version__}"
          f"   catboost {catboost.__version__}   torch {torch.__version__}")
    results.append(check("torch CUDA available", torch.cuda.is_available(),
                         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"))

    # xgboost 3.x wheels are built for CUDA 13 and silently fall back to CPU on this
    # 12.9 driver -- a warning, not an error, and a 14x slowdown. See docs/issues/001.
    xgb_cuda = xgboost.build_info().get("CUDA_VERSION")
    results.append(check("xgboost built for CUDA 12 (not 13)",
                         bool(xgb_cuda) and xgb_cuda[0] == 12,
                         f"CUDA_VERSION {xgb_cuda} -- pin xgboost<3.0 if this fails"))

    # nvidia-nccl-cu12 and -cu13 share one install directory, so removing either can
    # delete the other's library and break `import torch` with no package-level sign.
    import glob
    nccl = glob.glob(str(PROJECT_VENV / "nvidia/nccl/lib/libnccl.so*"))
    results.append(check("libnccl present on disk", bool(nccl),
                         "uv pip install --reinstall-package nvidia-nccl-cu12" if not nccl else ""))
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"         VRAM {vram:.1f} GB")

    # pandas >= 3.0 changed astype(str) to preserve NA. string_levels() works around it;
    # this asserts the workaround, not the pandas version.
    print("\n== the pandas target-encoding trap ==")
    probe = pd.DataFrame({"c": [1.0, np.nan, 2.0]})
    naive_cover = probe["c"].astype(str).to_frame("lv").groupby("lv").size().sum()
    fixed_cover = string_levels(probe, ["c"]).groupby("c").size().sum()
    print(f"         naive astype(str) covers {naive_cover}/3 rows"
          f"   (pandas {pd.__version__})")
    results.append(check("string_levels() gives missing its own level", fixed_cover == 3,
                         f"covers {fixed_cover}/3"))

    print("\n== data ==")
    train, test = load_train(), load_test()
    results.append(check("train shape", train.shape == (C.N_TRAIN, 14), str(train.shape)))
    results.append(check("test shape", test.shape == (C.N_TEST, 13), str(test.shape)))
    results.append(check("feature columns match config",
                         set(C.FEATURES) == set(train.columns) - {C.ID_COL, C.TARGET}))
    results.append(check("ids unique", train[C.ID_COL].is_unique and test[C.ID_COL].is_unique))

    y = train[C.TARGET].to_numpy(np.int8)
    rate = float(y.mean())
    results.append(check("positive rate ~0.7094", abs(rate - 0.7094) < 1e-3, f"{rate:.6f}"))

    print("\n== the budget identity ==")
    # min_count forces NaN unless ALL three parts are present; a plain sum() would
    # skip NaNs and silently weaken the check to "daily is present".
    parts = train[C.BUDGET_PARTS].sum(axis=1, min_count=len(C.BUDGET_PARTS))
    other = train[C.BUDGET_TOTAL] - parts
    complete = other.notna()
    violations = int((other[complete] < -1e-9).sum())
    results.append(check("daily >= social+gaming+work, zero violations", violations == 0,
                         f"{violations} of {int(complete.sum()):,} complete rows"))
    auc_other = roc_auc_score(y[complete.to_numpy()], other[complete].to_numpy())
    results.append(check("other_screen standalone AUC ~0.765", 0.70 < auc_other < 0.82,
                         f"{auc_other:.4f}"))

    print("\n== missingness ==")
    n_missing = train[C.FEATURES].isna().sum(axis=1).to_numpy()
    auc_missing = roc_auc_score(y, n_missing)
    results.append(check("n_missing carries no target signal", abs(auc_missing - 0.5) < 0.01,
                         f"AUC {auc_missing:.4f}"))
    tr_rate = train[C.FEATURES].isna().mean()
    te_rate = test[C.FEATURES].isna().mean()
    max_gap = float((te_rate - tr_rate).abs().max())
    results.append(check("train/test missing rates differ (expected)", max_gap > 0.01,
                         f"max gap {100 * max_gap:.2f} pp"))

    print("\n== folds ==")
    folds = load_folds()
    counts = np.bincount(folds, minlength=C.N_SPLITS)
    results.append(check("folds cover every train row", folds.min() >= 0 and len(folds) == C.N_TRAIN))
    results.append(check("folds balanced", counts.max() - counts.min() <= 1, str(counts.tolist())))

    print("\n== cardinalities (drive max_bin) ==")
    card = train[C.NUM_COLS].nunique().sort_values(ascending=False)
    for col, n in card.items():
        print(f"         {col:<28} {n:>6,}")
    results.append(check("max distinct <= 2047 (max_bin target)", int(card.max()) <= 2047,
                         f"max {int(card.max()):,}"))

    print()
    if all(results):
        print(f"PASS — {len(results)} checks")
    else:
        raise SystemExit(f"FAILED — {sum(not r for r in results)} of {len(results)} checks")


if __name__ == "__main__":
    main()
