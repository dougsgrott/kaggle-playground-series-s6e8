# 002 — `src/s6e8/features.py`, the measured-positive set

**Status:** done 2026-08-26
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

- [x] **Stringified target + frequency encoding**, all 12 columns, smoothing 10, nested 5×5 —
      **+0.0023 CV / +0.0017 LB**
- [x] **Impute alongside, never replacing** the raw NaN columns — **+0.0012**
- [x] **CatBoost's own ordered target statistics** on raw string levels — **+0.0004** over the
      hand-rolled encoder, for one argument
- [x] **Transductive frequency counts** over train+test (987,671 rows) — **+0.0003**
- [x] **`other_screen` residual** and the composition block — **+0.0005**
- [x] **Decimal lattice**: `frac_c = v − floor(v)`, `d1_c = floor(v*10) % 10` — **+0.0001**
- [x] Decimal-place counts read from **raw CSV text** (`data.py::read_decimal_places` already
      exists) — part of the structural-artifacts block
- [x] The three **categorical** `__isna` flags only

## Outcome

`src/s6e8/features.py` + `scripts/ablate_features.py`. Cumulative ablation on the frozen folds,
**+0.003903 over raw** (0.964007 → 0.967910). Full table, per-block null sd and the corpus
comparison: [`../experiments.md`](../experiments.md#002--feature-blocks-cumulative-ablation).
Raw cells: [`../ablation_002.json`](../ablation_002.json),
[`../ablation_002_decimals.json`](../ablation_002_decimals.json).

| block | Δ | verdict | vs corpus |
|---|---:|---|---:|
| `lattice` | **+0.001609** | PAYS | **16× the published +0.0001** |
| `te` | +0.000740 | PAYS | 0.32× |
| `budget` | +0.000626 | PAYS | 1.3× |
| `freq` | +0.000609 | PAYS | 2.0× |
| `impute` | +0.000193 | PAYS | 0.16× |
| `decimals` | +0.000112 | PAYS | — |
| `cat_isna` | −0.000007 | **UNRESOLVED** | 0× |

Shipped as the `xgb_features` member. `cat_isna` is excluded — unresolved is not a reason to add
three columns to every member downstream.

## What it bought beyond a number

1. **The plan's ordering was wrong, and by a lot.** The decimal lattice was ranked fifth of six at
   +0.0001 and is the biggest block by 2.6×. It is a value-encoding mechanism — `frac` and `d1`
   let a tree separate individual printed values, the same job `max_bin` and target encoding do.
   Conversely `te` (+0.000740 vs +0.0023) and `impute` (+0.000193 vs +0.0012) read low **because
   they are substitutes**, not because they failed: by the time they are added, `lattice` and
   `freq` have taken most of the exact-value signal. Priced first they would rank far higher.
   This is the same substitution the corpus found between `max_bin` and value encoding, and it is
   the concrete reason published deltas must never be summed.
2. **The gate from issue 003 is not portable, and this issue is what proved it.** σ_model spans
   **7×** across the seven feature sets measured here (0.000005 → 0.000032) and is not monotonic
   in column count. A flat 0.00011 would have been far too strict at the raw end. Every
   comparison now carries its own null sd, `sqrt(σa²/na + σb²/nb)`.
3. **`decimals` was a CV↑/LB↓ trap and was defused, not accepted.** `read_decimal_places` returns
   NaN where the field is missing, and that mask is bit-identical to raw missingness — so the
   obvious version of the block is a numeric missingness flag with a decimal bit stapled on. But
   the halves separate: the 1dp/2dp rate does *not* shift train-to-test (|z| < 1.7 everywhere)
   while it does predict `y` (P(y|1dp) 0.6821 vs P(y|2dp) 0.7121 on `daily_screen_time_hours`).
   Mode-filling keeps the bit and drops the channel; **84% of the gain survived**, so it was the
   generator's printed-precision artifact rather than the train/test shift.
4. **A third verdict earned its place.** `cat_isna` came back −0.000007 against a 1.8e-05 null sd
   over three seeds per arm. Not a win, not a loss — the corpus's +0.0001 simply does not
   reproduce here, and recording that as "skip" would have been as wrong as recording it as
   "build".

## Built along the way

| what | where |
|---|---|
| six static blocks + `NestedTargetEncoder`, with a versioned on-disk cache | `src/s6e8/features.py` |
| cumulative ablation runner, three verdicts, per-arm σ | `scripts/ablate_features.py` |
| `xgb_features` member (early stopping back, cap raised to 6000) | `src/s6e8/models.py` |

Correctness notes worth keeping: `bd_resid` reproduces the corpus standalone AUC exactly (0.7648
vs 0.765) and uses `min_count`, without which the residual silently becomes "daily minus whatever
was present"; the frequency block's missing values form their own 79,295-count level rather than
mapping to 0, which the source notebook's `astype(str)` would have got wrong on pandas 3.x; the
encoder's leakage control shows train-row AUC 0.8770 against valid-row 0.8776, the right sign;
and the lattice uses `rint(v*100)` because `daily_screen_time_hours` and `weekend_screen_time`
are not 2-decimal quantised, so `floor(v*10)` would misread 1.8 as 1.7999.

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

## Exit criterion — met

Ablation table in [`../experiments.md`](../experiments.md), every row judged against its **own**
measured null sd rather than a borrowed constant, and the one `UNRESOLVED` block recorded as
neither a win nor a loss.

## Next

Phase 2 — issue 007 (XGBoost GPU members across feature views). `xgb_features` is registered and
ready; the first thing it should produce is a real OOF and a second CV→LB point, since the line
currently rests on one observation at CV 0.9649.
