# 002 — `src/s6e8/features.py`, the measured-positive set

**Status:** open — **next**
**Phase:** 1 — the real work
**Do third** — after 004 and 003.

## The gate, now that issue 003 has measured it

**σ_delta = 0.000055; a block must clear 2σ = 0.00011 to count.** That is the null spread of an
A/B delta on the frozen partition with both arms re-seeded — two *identical* configs were observed
0.000101 apart.

Consequences for the list below, and they are not cosmetic:

- The **top four blocks are safe** — +0.0003 to +0.0023 is 5×–42× the gate.
- The **decimal lattice at +0.0001 is under the gate.** So are the `__missing__` level (+0.0001),
  10 encoding folds (+0.0001), constraint geometry (+0.00006) and regime-aware stacking
  (+0.00004). Those corpus figures were gated at 1× a floor of the same size, which is a coin
  flip. Build them last, and only price them with the multi-seed protocol below — or accept them
  on structural grounds without claiming a measured gain.
- **To resolve a block near the gate, average model seeds, not partitions.** The model seed
  carries twice the variance of the partition seed here (0.000039 vs 0.000019), so σ_delta falls
  as √n over seeds: 0.000032 at n=3, halving the gate to 0.000064. At ~2.6 min per 5-fold XGBoost
  run that is ~8 min per arm — cheap enough to spend on any block whose expected gain is ≤0.0002,
  and wasted on the ones above +0.0005.
- **Never read the per-fold spread as uncertainty.** It runs ~0.0012–0.0017 on this partition and
  is mostly fold 3 being reproducibly easier (+0.00097) and fold 0 harder (−0.00061).

## Scope, in descending measured value

Build only these, in this order, ablating each against **2σ_delta = 0.00011**:

- [ ] **Stringified target + frequency encoding**, all 12 columns, smoothing 10, nested 5×5 —
      **+0.0023 CV / +0.0017 LB**
- [ ] **Impute alongside, never replacing** the raw NaN columns — **+0.0012**
- [ ] **CatBoost's own ordered target statistics** on raw string levels — **+0.0004** over the
      hand-rolled encoder, for one argument
- [ ] **Transductive frequency counts** over train+test (987,671 rows) — **+0.0003**
- [ ] **`other_screen` residual** and the composition block — **+0.0005**
- [ ] **Decimal lattice**: `frac_c = v − floor(v)`, `d1_c = floor(v*10) % 10` — **+0.0001**
- [ ] Decimal-place counts read from **raw CSV text** (`data.py::read_decimal_places` already
      exists) — part of the structural-artifacts block
- [ ] The three **categorical** `__isna` flags only

## Two things to get right the first time

### 1. Every encoder key goes through `src/s6e8/data.py::string_levels`

**pandas 3.0.5 is installed, so the trap is live.** On 3.x the new `str` dtype preserves NA, so
`astype(str)` lets `groupby` silently drop every missing row from the level statistics and
`fillna(prior)` quietly hands them the base rate. No error, no warning — `scripts/check_env.py`
measures it directly: naive `astype(str)` covers 2 of 3 probe rows.

`string_levels()` routes through `object` and fills first, which is correct on both 2.x and 3.x.
Add a coverage assertion at every `groupby`:

```python
assert grouped.size().sum() == len(df)
```

### 2. Measure `max_bin` *after* the encodings land

`max_bin` and value encoding are **substitutes, not additive** — they do the same job of letting
the model tell individual values apart. Measured separately at +0.0024 and +0.0018; doing both
gives +0.0022, just under **60% of the sum**. Our own Phase 0 probe saw +0.00142 for
`max_bin` 255→2047 on raw columns, and that will shrink once target encoding is in.

So: land the encodings, then sweep `max_bin` against the local floor. Never quote the two
published gains added together.

## Do not build

Each was measured negative — the reason is recorded so it is not re-litigated:

`pseudo-labelling` (−0.0034) · `denoising-autoencoder features` (−0.00071) · `pairwise TE`
(−0.00040) · `multi-resolution TE` (−0.00033) · `monotone constraints` (−0.0003) · `TE smoothing
50/200` (−0.00006 / −0.00030) · `depth 9–13` (to −0.0011) · `the original 7,500-row dataset`
(−0.0001) · `numeric is_missing flags` (−0.00001, and they identify the train/test split) ·
`identity/digit categoricals` (below baseline — the continuous fractional part wins) ·
`behavioural ratios, sleep_deficit, weekend-ratio, total_weekly_screen_time`.

## Sources

- `analysis/nb_clean/tomasa2__s6e8-what-moved-the-score-and-what-didn-t.py` — the ablation of
  record, and the nested `build_enc` scheme
- `analysis/nb_clean/dariushafshar__s6e8-what-actually-helps.py` — decimal places from raw text,
  the leakage audit with a control
- `analysis/nb_clean/najiama__single-lgbm-model-lb-0-96990-cv-0-96862.py` — transductive counts

## Exit criterion

An ablation table in `docs/experiments.md` with every row expressed in multiples of the measured
σ_delta = 0.000055 and judged against 2σ = 0.00011, including the blocks that did not pay — and
the blocks that came back *unresolved*, which is a third verdict this data will produce and which
must not be recorded as either a win or a loss.

## Next

Phase 2 — issue 007 (XGBoost GPU members across feature views) first, since it is the cheapest way
to price the new blocks.
