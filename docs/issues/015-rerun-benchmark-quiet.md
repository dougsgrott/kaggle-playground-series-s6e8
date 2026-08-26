# 015 — Re-run the benchmark on a quiet machine

**Status:** open
**Phase:** loose end — clear early
**Estimate:** ~20 min of wall clock, unattended

## Why

`scripts/benchmark.py` was run at **load average ~45 on 20 cores**. Everything it reports is a
**load-dependent floor, not a constant**, and the Phase 2 schedule inherits whatever error is in
those numbers:

| member | measured 1 fold | scaled 5 folds @ 3,000 rounds |
|---|---:|---:|
| XGBoost GPU | 27.2 s (400 rounds) | ~17 min |
| CatBoost GPU | 102.8 s (200) | ~2.1 h |
| LightGBM `max_bin=2047` | 139.5 s (200) | ~2.9 h |

The GPU numbers should barely move. The **CPU-bound LightGBM number could improve several-fold**,
and that is the one deciding how many LightGBM members Phase 2 can afford.

## Scope

- [ ] Check `MemAvailable` and `/proc/loadavg` first; only run when load is near idle
- [ ] `uv run python scripts/benchmark.py`
- [ ] Sweep `S6E8_THREADS` (1, 2, 4, 8) on the quiet box — the optimum measured under load was 2,
      and that is a contention artifact, not a property of the data
- [ ] Update the compute table in `docs/ROADMAP.md` Phase 2 and the table in `issues/001`
- [ ] Revisit the "budget LightGBM and CatBoost members deliberately" guidance if the CPU
      families get materially cheaper

## Note

Record the load average alongside the timings, so the next reader can tell which regime a number
came from. A timing with no load figure attached is not reproducible on this box.

## Next

Feeds the Phase 2 member budget in `docs/ROADMAP.md`.
