# Experiment ledger (append-only)

One row per measured change. Negative results included — they are the point.

**Rules**
- Every delta is expressed in multiples of the **noise floor** (≈ 0.00005 = std of repeated-5-fold
  means over 3 partition seeds). A delta under 1× the floor has measured nothing.
- The per-fold AUC range inside a single run is ~10× the noise floor and must never be quoted as
  uncertainty.
- Record the fold seed, the member set, and whether weights were fitted naively or nested.

| date | id | change | fold AUC | Δ vs base | × floor | LB | notes |
|---|---|---|---|---|---|---|---|
| 2026-08-26 | 001a | LightGBM baseline, raw 12 cols, leaves 63, `max_bin` 255, 200 rounds | 0.95852 | — | — | — | fold 0 only, undertrained; the reference point for the two rows below |
| 2026-08-26 | 001b | **capacity**: `num_leaves` 63 → 255 | 0.96161 | **+0.00309** | 62× | — | confirms the corpus finding that capacity dominates feature engineering |
| 2026-08-26 | 001c | **`max_bin` 255 → 2047** (at leaves 255) | 0.96303 | **+0.00142** | 28× | — | vs 001b; corpus reports +0.0024 on raw columns, same direction and order |
| 2026-08-26 | 001d | XGBoost GPU, depth 7, 400 rounds | 0.95894 | — | — | — | 27.2 s/fold — the cheapest member by 4–8× |
| 2026-08-26 | 001e | CatBoost GPU, depth 6, native cats, 200 rounds | 0.94124 | — | — | — | undertrained at lr 0.05; timing reference only |

**These are single-fold, deliberately undertrained probes from `scripts/benchmark.py`, not OOF
numbers.** They exist to confirm the two biggest published levers reproduce on our own data before
Phase 2 commits to them, and they do: capacity first, then `max_bin`. Treat the magnitudes as
indicative — 200 rounds at lr 0.05 is far from converged, and `max_bin` and value encoding are
substitutes, so the +0.00142 will shrink once target/frequency encoding is added.

The proper noise floor (repeated 5-fold across 3 partition seeds) has **not** been measured
locally yet — that is issue 003. The `× floor` column above uses the corpus value of 0.00005.

---

## Prior calibration points harvested from the corpus

Not our measurements — anchors to sanity-check ours against. Sources in
[`data-notes.md`](data-notes.md) and the cited `analysis/nb_clean/` files.

### CV → LB

| source | CV / OOF | public LB | offset |
|---|---:|---:|---:|
| `rugvedbane__s6e8-13-fe-features-xgboost-optuna-0-96602` | — | 0.96602 | — |
| discussion 735000, untuned XGB + poly FE | 0.964929 | 0.96646 | +0.00153 |
| discussion 736595, XGB 4 features 10-fold | 0.967663 | 0.96844 | +0.00078 |
| `najiama__single-lgbm-model-lb-0-96990-cv-0-96862` | 0.96862 | 0.96990 | +0.00128 |
| `tomasa2__*` 5-model library | 0.968838 | 0.97014 | +0.00130 |
| `tomasa2__*` 40-member stack | 0.969635 | 0.97074 | +0.00110 |
| `dariushafshar__fork-this-*` nested honest stack | 0.969735 | 0.97083 | +0.00110 |
| `laymond__s6e8-elasticnet-*` | 0.970029 | 0.97108 | +0.00105 |

**The offset is a line, not a constant**: it decays from +0.00150 at CV 0.9660 to +0.00109 at
CV 0.9696, `corr(CV, offset) = −0.99`. Test predictions average 5 fold-models while OOF comes from
1, and a large stack has already averaged that variance away.

### Measured feature/technique deltas (from `tomasa2__s6e8-what-moved-the-score-and-what-didn-t.py`)

Noise floor in that notebook: **0.00005**.

| idea | Δ | verdict |
|---|---:|---|
| stringified TE + freq, all 12 columns | **+0.0023 CV / +0.0017 LB** | build |
| imputed columns *alongside* the NaNs | +0.0012 | build |
| composition / ratio features | +0.0005 | build |
| CatBoost native ordered target statistics | +0.0004 | build |
| logit stack over the library | +0.0004 | build |
| transductive frequency encoding | +0.00032 solo | build |
| lower learning rate (depth 5 @ 0.01) | +0.0002 | build |
| 10 encoding folds instead of 5 | +0.0001 | build |
| decimal lattice (`frac`, first digit) | +0.0001 | build |
| explicit `"__missing__"` level | +0.0001 on pandas ≥ 3.0 | build |
| missingness augmentation on the NN | +0.00008 solo / +0.00005 LB | build |
| constraint-geometry features | +0.00006 | marginal |
| regime-aware stacking | +0.00004 | marginal |
| NA-indicator features | −0.00001 | skip |
| pairwise TE | −0.00040 | skip |
| multi-resolution TE | −0.00033 | skip |
| TE smoothing 50 / 200 | −0.00006 / −0.00030 | skip |
| rank / rank-gauss at the stacker | −0.00013 / −0.00002 | skip |
| features from the 7,500-row original | −0.00003 | skip |
| **denoising-autoencoder features** | **−0.00071** | skip |
| monotone constraints | −0.0003 | skip |
| concatenating the original dataset | −0.0001 | skip |
| tree depth 9–13 | to −0.0011 | skip |
| naive mean of a 12-model library | −0.0012 | skip |
| **pseudo-labelling confident test rows** | **−0.0034** | skip |

### `max_bin` × value-encoding (they are substitutes, not additive)

Same folds, same LightGBM params, only columns and `max_bin` change:

| columns | `max_bin`=255 | `max_bin`=2047 | gain |
|---|---:|---:|---:|
| raw 12 | 0.96431 | 0.96672 | **+0.0024** |
| 18 (+ leftover + interval) | 0.96541 | 0.96746 | +0.0020 |
| 21 (+ frequency encoding on 3 columns) | 0.96717 | 0.96767 | **+0.0005** |

Frequency columns are worth +0.0018 at 255 and +0.0002 at 2047. Measured separately +0.0024 and
+0.0018; doing both gives +0.0022 — just under 60% of the sum.

### Best published single models

| model | OOF | LB |
|---|---:|---:|
| Lookup-Transformer blend | 0.96872 | 0.97041 |
| RealMLP (hand-written PyTorch) | — | 0.97014 / 0.97009 |
| LightGBM + exact-value TE | 0.96862 | 0.96990 |
| LightGBM + TE, 3 seeds, no blend | — | 0.96949 |
| CatBoost + constraint feature (11-fold) | 0.96832 | — |
