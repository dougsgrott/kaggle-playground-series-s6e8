"""Process setup that must happen before lightgbm/xgboost are imported.

Both size their OpenMP pools at load time, so `configure_threads()` has to run first.
This box has 20 cores but has been observed at load average ~50, where extra threads
spin and contend: the same XGBoost fit took 2.0 s at 1 thread and 18.0 s at 4.
See docs/issues/001-phase-0-ground-truth.md.
"""
from __future__ import annotations

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def configure_threads() -> int:
    """Cap the OpenMP pools. Call before importing any modelling library."""
    threads = os.environ.setdefault("S6E8_THREADS", "2")
    for var in _THREAD_VARS:
        os.environ.setdefault(var, threads)
    return int(threads)


def n_threads() -> int:
    return int(os.environ.get("S6E8_THREADS", "2"))


def assert_xgboost_gpu() -> str:
    """Fail loudly if xgboost would silently fall back to CPU.

    3.x wheels are built against CUDA 13; this WSL driver provides CUDA 12.9, so a 3.x
    build prints a *warning* and trains on CPU 14x slower. A silent slowdown is worse
    than a crash, so every GPU member asserts the device it asked for.
    """
    import xgboost as xgb

    cuda = xgb.build_info().get("CUDA_VERSION")
    if not cuda or cuda[0] != 12:
        raise RuntimeError(
            f"xgboost {xgb.__version__} is built for CUDA {cuda}; this driver is 12.9, so "
            "device='cuda' would silently fall back to CPU. Pin xgboost>=2.1,<3.0."
        )
    return f"xgboost {xgb.__version__} (CUDA {cuda[0]}.{cuda[1]})"
