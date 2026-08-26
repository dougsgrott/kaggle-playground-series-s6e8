# 001 — Phase 0: environment, data, frozen folds

**Status:** done (2026-08-26)
**Phase:** 0

## Scope

- [x] ML dependencies in `pyproject.toml`; `uv sync`
- [x] Competition data downloaded to `data/`
- [x] Repo skeleton (`docs/`, `src/s6e8/`, `scripts/`, `data/`, `oof/`, `submissions/`)
- [x] Frozen fold contract materialised to `data/folds.npy`
- [x] `scripts/check_env.py` gate passing
- [x] Real timing benchmark to replace the estimated Phase 2 schedule

## Outcome

`uv run python scripts/check_env.py` → **PASS, 14 checks**.

```
pandas 3.0.5   numpy 2.5.2   lightgbm 4.7.0   xgboost 3.4.1   catboost 1.2.10   torch 2.6.0+cu124
CUDA: NVIDIA GeForce RTX 4050 Laptop GPU, 6.0 GB
train (691369, 14)   test (296302, 13)   positive rate 0.709424
folds sha256 9571f18b6ba175d426593d5576d4846d30f00e015cae09d657b3a53437bee824
```

## What the setup established

**The frozen partition.** `data/folds.npy`, sizes `[138274, 138274, 138274, 138274, 138273]`,
per-fold positive rates all 0.70942. The sha256 above is the contract: any member whose OOF was
not built on this partition cannot enter a stack, and the digest is what proves it.

**The budget identity holds exactly.** Zero violations on the 421,427 rows where all four columns
are present; `other_screen` standalone **AUC 0.7649**, reproducing the corpus figure of 0.765.

A subtlety worth recording: `df[PARTS].sum(axis=1)` skips NaNs by default, which silently weakens
the check to "`daily` is present" and inflates the complete-row count from 421,427 to 595,515.
`min_count=3` is required. The first version of `check_env.py` had this bug and passed anyway —
with a wrong AUC of 0.7116.

**pandas 3.0.5 is installed, so the target-encoding trap is live, not hypothetical.** The gate
measures it directly: naive `astype(str)` covers 2 of 3 probe rows, `string_levels()` covers 3 of
3. On pandas 3.x the new `str` dtype preserves NA, so `groupby` silently drops every missing row
from the level statistics and `fillna(prior)` quietly hands them the base rate — no error, no
warning. `src/s6e8/data.py::string_levels` is the only sanctioned way to build encoder keys.

**Cardinalities correct a corpus claim.** The widest column is `weekend_screen_time` at **1,437**
distinct values, not `daily_screen_time_hours` (1,389). Both sit above 1,400 and together account
for 91% of the `max_bin` gain, so the operational conclusion — `max_bin = 2047` — is unchanged.

## The memory constraint is tighter than the spec sheet

The first benchmark attempt died producing **zero output**. The cause was two-part and both halves
are worth keeping:

1. **stdout was block-buffered to a file**, so every `print` before the kill was lost with the
   buffer. A background run that dies looks identical to one that never started. Always run
   long jobs with `PYTHONUNBUFFERED=1` and flush explicitly.
2. **The machine had ~2 GB free, not 7.** `free -m` during Phase 0: 7,801 MB total, 5,691 MB used,
   **1,987 MB available**, with 1.3 GB of the 2 GB swap already consumed. Two unrelated `pytest`
   processes held ~2 GB and the Claude processes ~1.5 GB.

The data itself is not the problem — the full 12-column frame as float32 is **271 MB RSS**. What
kills a run is a careless copy: `pd.get_dummies(X).astype(np.float32)` materialises two additional
full-width frames, and the fusion layer is far worse.

Operational rules that follow, now in the roadmap risk register:

- Load only the needed columns, cast to float32 on read.
- Never one-hot the full frame; pass `enable_categorical=True` / native `cat_features` instead.
- `del` and `gc.collect()` between members; the benchmark stamps RSS at every step.
- Check `MemAvailable` in `/proc/meminfo` before launching anything long.

## Two environment traps that would each have cost days

### 1. XGBoost 3.x silently trains on CPU

`torch.cuda.is_available()` returns `True`, `nvidia-smi` is healthy, and CatBoost's GPU works —
yet XGBoost printed:

```
WARNING: No visible GPU is found, setting device to CPU.
WARNING: Device is changed from GPU to CPU as we couldn't find any available GPU on the system.
```

Cause: `xgboost==3.4.1` reports `build_info()["CUDA_VERSION"] == [13, 3]`, while this WSL driver
provides **CUDA 12.9**. A CUDA 13 binary cannot bind a 12.9 driver, so it falls back — with a
warning that is easy to filter out and no error at all.

Fix: pin `xgboost>=2.1,<3.0` in `pyproject.toml`. `xgboost==2.1.4` reports `CUDA_VERSION [12, 8]`
and trains on the GPU: **50 trees on 200,000 rows in 2.5 s**, versus 34.9 s on the CPU fallback.
No `LD_LIBRARY_PATH` adjustment is needed — the version mismatch was the entire problem.

**Any member script must assert it actually got the device it asked for.** A silent 14× slowdown
is worse than a crash.

### 2. Thread oversubscription — more threads made it 9× slower

The box has 20 cores but ran at **load average ~50** (unrelated work). Same XGBoost fit,
100,000 rows, 50 trees:

| threads | time |
|---:|---:|
| 1 | **2.0 s** |
| 4 | 18.0 s |
| 20 | did not finish in 400 s |

LightGBM shows the same shape (n_jobs=2 → 1.9 s, n_jobs=4 → 11.2 s). With
`OMP_NUM_THREADS=1` exported *before import*, XGBoost is fast at every `n_jobs` value — the env
var caps the real OpenMP pool regardless of what the estimator asks for.

This is what made the first two benchmark attempts look like hangs: they were not hung, they were
thrashing, and the buffered stdout hid it.

**Rule**: set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` before importing
lightgbm or xgboost, and pass `n_jobs` to match. `scripts/benchmark.py` does this at the top via
`S6E8_THREADS` (default 2). Prefer the GPU where a family supports it — XGBoost and CatBoost both
do, and neither is affected by the CPU contention.

### 3. Pinning XGBoost silently broke torch

Pinning `xgboost<3.0` uninstalled `nvidia-nccl-cu13` (an xgboost 3.x dependency). Both
`nvidia-nccl-cu12` and `-cu13` install into the **same** `nvidia/nccl/lib/` directory, so removing
one deleted the other's `libnccl.so.2`. `uv pip list` still showed `nvidia-nccl-cu12 2.21.5` as
installed — the package metadata was intact, the file was gone — and `import torch` then failed
with `ImportError: libnccl.so.2: cannot open shared object file`.

Fix: `uv pip install --reinstall-package nvidia-nccl-cu12 nvidia-nccl-cu12==2.21.5`.

`scripts/check_env.py` now checks the library is on disk, not just that the package is listed.
The gate is at **16 checks** and catches all three traps above.

## Measured cost per member

One fold (553,095 train / 138,274 valid), raw 12 columns, load average ~45:

| member | config | rounds | 1 fold | fold AUC | peak RSS |
|---|---|---:|---:|---:|---:|
| XGBoost GPU | depth 7 | 400 | **27.2 s** | 0.95894 | 554 MB |
| LightGBM CPU (2 thr) | leaves 63, `max_bin` 255 | 200 | 85.1 s | 0.95852 | 430 MB |
| LightGBM CPU (2 thr) | leaves 255, `max_bin` 255 | 200 | 94.7 s | 0.96161 | 430 MB |
| LightGBM CPU (2 thr) | leaves 255, `max_bin` 2047 | 200 | 139.5 s | **0.96303** | 430 MB |
| CatBoost GPU (native cats) | depth 6 | 200 | 102.8 s | 0.94124 | 1,076 MB |
| CatBoost CPU (4 thr) | depth 6 | 200 | 204.3 s | 0.94097 | 1,076 MB |

**XGBoost GPU is 4–8× cheaper than any other family here** and is immune to the CPU contention.
It should carry the feature-view diversity in Phase 2; LightGBM and CatBoost members should be
budgeted deliberately rather than spawned freely.

Two incidental findings worth keeping: CatBoost's `eval_metric="AUC"` is **not implemented on
GPU** and forces a slow path (it prints `Default metric period is 5 because AUC is/are not
implemented for GPU`) — a 400-iteration run exceeded 500 s before this was removed. And LightGBM
wants `force_row_wise=True` here to skip a costly auto-detect pass.

### The two published levers reproduce on our own data

| change | fold AUC | Δ |
|---|---:|---:|
| leaves 63 → 255 (capacity) | 0.95852 → 0.96161 | **+0.00309** |
| `max_bin` 255 → 2047 at leaves 255 | 0.96161 → 0.96303 | **+0.00142** |

Same direction and order of magnitude as the corpus (which reports +0.0024 for `max_bin` on raw
columns), and it confirms the ordering the roadmap depends on: **capacity first, then `max_bin`,
then features**. Both numbers are single-fold and undertrained, so treat them as confirmation of
direction, not as final magnitudes — and remember `max_bin` and value encoding are substitutes,
so the +0.00142 will shrink once target encoding lands.

## A note on the three "hangs"

Three benchmark runs appeared to hang and died with no output. None of them was an OOM:

- Runs 1 and 2 were **thread thrashing** (20 OpenMP threads on a box at load 50), compounded by
  **block-buffered stdout** that discarded every `print` when the process was killed.
- Run 3 died because a process backgrounded with `nohup ... &` *inside a normal tool call* is
  reaped when that call's shell is cleaned up. Long jobs must use the harness's own background
  mechanism, not `&`.

Peak RSS never exceeded 1.1 GB for any single member, so memory is a fusion-layer problem, not a
member-training one.

## Next

Issue 002 (`src/s6e8/features.py`) and 003 (noise floor + repeated-CV harness).
