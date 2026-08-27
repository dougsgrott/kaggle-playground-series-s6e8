# Experiment ledger (append-only)

One row per measured change. Negative results included — they are the point.

**Rules**
- Every delta is expressed in multiples of **σ_delta = 0.000055**, measured on this box
  (issue 003, [`noise_floor.json`](noise_floor.json)). That is the null spread of an A-vs-B delta
  on the frozen partition with both arms re-seeded — the right yardstick here, because every row
  below holds the partition at seed 42 and varies only the config.
- **The gate is 2σ_delta = 0.00011.** Two *identical* configs were observed 0.000101 apart, so a
  delta under ~1e-4 has measured nothing regardless of how many folds agree on its sign.
- Comparing against a CV computed on a **different** split is a different question with a smaller
  floor: σ_partition = 0.000019.
- **Buy precision with model seeds, not partitions.** σ_delta falls as √n when both arms average n
  model seeds — 0.000032 at n=3. Repeating CV across partition seeds does not help; the partition
  contributes half as much variance as the model seed.
- The per-fold AUC range inside a single run is **22×** σ_delta and must never be quoted as
  uncertainty. On the frozen partition it is mostly fold 3 being genuinely easier (+0.00097) and
  fold 0 harder (−0.00061) — a fixed property of the split, reproduced across six model seeds.
- Record the fold seed, the member set, and whether weights were fitted naively or nested.

| date | id | change | fold AUC | Δ vs base | × floor | LB | notes |
|---|---|---|---|---|---|---|---|
| 2026-08-26 | 001a | LightGBM baseline, raw 12 cols, leaves 63, `max_bin` 255, 200 rounds | 0.95852 | — | — | — | fold 0 only, undertrained; the reference point for the two rows below |
| 2026-08-26 | 001b | **capacity**: `num_leaves` 63 → 255 | 0.96161 | **+0.00309** | 56× | — | confirms the corpus finding that capacity dominates feature engineering |
| 2026-08-26 | 001c | **`max_bin` 255 → 2047** (at leaves 255) | 0.96303 | **+0.00142** | 26× | — | vs 001b; corpus reports +0.0024 on raw columns, same direction and order |
| 2026-08-26 | 001d | XGBoost GPU, depth 7, 400 rounds | 0.95894 | — | — | — | 27.2 s/fold — the cheapest member by 4–8× |
| 2026-08-26 | 001e | CatBoost GPU, depth 6, native cats, 200 rounds | 0.94124 | — | — | — | undertrained at lr 0.05; timing reference only |
| 2026-08-26 | 004 | **`xgb_baseline`** — `cdeotte__simple-xgb-starter.py` ported verbatim to the frozen folds | **0.964869** (pooled OOF) | — | — | **0.96640** | offset +0.001531, within 1e-4 of the corpus line; ~rank 1154/2987 |

**Row 004 is a real pooled OOF number. Rows 001a–001e are single-fold, deliberately undertrained
probes from `scripts/benchmark.py`, not OOF numbers.** They exist to confirm the two biggest published levers reproduce on our own data before
Phase 2 commits to them, and they do: capacity first, then `max_bin`. Treat the magnitudes as
indicative — 200 rounds at lr 0.05 is far from converged, and `max_bin` and value encoding are
substitutes, so the +0.00142 will shrink once target/frequency encoding is added.

Rows 001a–001e ran on **fold 0**, which issue 003 later showed is the *hardest* fold of the frozen
partition by 0.00061. Their absolute levels therefore read low against any pooled number; their
deltas are unaffected, since both arms share the fold.

The `× floor` column is now divided by the **locally measured** σ_delta = 0.000055 (issue 003,
55 fits). The previous values used the corpus constant of 0.00005; the two changed rows moved
62×→56× and 28×→26×, which changes no verdict — both levers clear the gate by more than an order
of magnitude.

### 007 — XGBoost feature views, and the admission rule

Real members: early stopping, lr 0.035 (0.03 for `xgb_deep`), cap 6000, on the frozen folds.
Each exports the positional OOF contract. Matrix and blend: `scripts/member_matrix.py`.

| date | id | member | blocks | solo OOF | vs best | ρ to flagship | LB |
|---|---|---|---|---:|---:|---:|---:|
| 2026-08-26 | 007a | **`xgb_features`** | all six + `te` | **0.968616** | — | — | **0.96989** |
| 2026-08-26 | 007b | `xgb_te_only` | `raw`+`budget`+`freq`+`impute` + `te` | 0.968596 | −0.000020 | 0.9980 | — |
| 2026-08-26 | 007c | `xgb_deep` | all six + `te`, depth 10 | 0.968531 | −0.000085 | 0.9976 | — |
| 2026-08-26 | 007d | `xgb_no_te` | all six, **no** `te` | 0.967691 | −0.000925 | 0.9928 | — |
| 2026-08-26 | 004 | `xgb_baseline` | starter block | 0.964869 | −0.003747 | 0.9865 | 0.96640 |

#### The admission rule is backwards at this stack size

Nested-CV logit blend — weights fitted on 4 folds, scored on the 5th, never naively on the full
OOF (non-negotiable #3).

| added to `xgb_features` | ρ | corpus rule says | **actually adds** |
|---|---:|---|---:|
| `xgb_no_te` | 0.9928 | REJECT | **+0.000228** |
| `xgb_te_only` | 0.9980 | REJECT | +0.000135 |
| `xgb_baseline` | 0.9865 | ADMIT | +0.000108 |
| `xgb_deep` | 0.9976 | REJECT | +0.000100 |
| **all five** | — | 2 of 5 admitted | **+0.000334** → nested 0.968948 |

**Contribution is rank-ordered backwards by correlation.** The rule as written would have kept the
two least useful members and discarded the most useful one. For scale, +0.000334 from five members
is more than the corpus's entire 205-member fusion pyramid buys over a good honest stack (~0.0001).

The reconciliation is probably **stack size, not a broken corpus**: at 70 members a 0.99-correlated
addition really is redundant because something already covers that direction, while at five it
still buys variance reduction. Two observations support saturation rather than a linear rule — the
5th member adds only **+0.000016** on top of the 4th, and the pairwise contributions (0.000100 to
0.000228) are far smaller than their sum would suggest. **Do not discard members on ρ alone yet;
re-test as the zoo grows.** The strength half of the rule is untouched.

#### Feature views inside one algorithm barely decorrelate

Every pair sits at ρ 0.9859–0.9980, and the floor is set by the *weakest* member, not the most
structurally different one. Dropping target encoding entirely — a completely different mechanism —
only reaches ρ 0.9928. **Diversity has to come from other model families, not more XGBoost views.**
The corpus figure for GBDT→NN with value embeddings is ρ 0.974 with blend weight 0.22, which is a
different regime from anything reachable here. Consequence for Phase 2: stop adding XGBoost views,
and prioritise the Lookup-Transformer and RealMLP members over more tree variants.

#### Two corpus claims that did not reproduce

| claim | corpus | measured here |
|---|---:|---:|
| decimal lattice, measured *after* target encoding | +0.0001 | +0.00002 — **agrees**, see below |
| tree depth 9–13 | to −0.0011 | **−0.000085** |

**The lattice/TE ordering artifact, resolved.** Issue 002 reported the decimal lattice at
+0.001609, sixteen times its published value, and flagged that as the headline finding. `xgb_te_only`
prices the same pair in the opposite order and the picture inverts: dropping the lattice costs
**0.00002** when `te` is present, while dropping `te` costs **0.000925** when the lattice is
present. Target encoding dominates; the cumulative chain simply handed the lattice credit for
signal the two share because it was measured first.

This **reconciles issue 002 with the corpus** rather than contradicting it — the corpus measured the
lattice with TE already in place, which is exactly the +0.00002 condition. Both numbers are right;
neither is a standalone property of the block. **The general lesson is that no single-block delta
from a cumulative ablation is a property of the block** — it is a property of the block *and the
order*. Where two features are substitutes, whichever is measured first takes the credit.

`xgb_deep` at depth 10 costing only 0.000085 is the second non-reproduction, and it is why it was
run: "same algorithm, different params → weight 0.000" was worth one cheap test, and it also earned
+0.000100 in the blend rather than the predicted zero.

### 002 — feature blocks, cumulative ablation

Same frozen folds, same fixed-round XGBoost the noise floor was measured on (lr 0.10 x 900,
no early stopping, native categoricals). Cumulative: each row adds one block to the row above,
so every delta answers *does this block pay on top of what we already have?* Raw numbers:
[`ablation_002.json`](ablation_002.json), [`ablation_002_decimals.json`](ablation_002_decimals.json).

| date | id | block | cols | pooled OOF | Δ vs prev | seeds | null sd | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-08-26 | 002a | `raw` — the 12 columns, native categoricals | 12 | 0.964007 | baseline | 3 | — | — |
| 2026-08-26 | 002b | **`budget`** — accounting residual + composition | 21 | 0.964633 | **+0.000626** | 3 | 1.0e-05 | **PAYS** |
| 2026-08-26 | 002c | `cat_isna` — the 3 categorical missing flags | 24 | 0.964626 | −0.000007 | 3 | 1.8e-05 | **UNRESOLVED** |
| 2026-08-26 | 002d | **`lattice`** — `frac` + `d1` on 6 fractional columns | 36 | 0.966235 | **+0.001609** | 3 | 2.4e-05 | **PAYS** |
| 2026-08-26 | 002e | **`decimals`** — printed decimal places, mode-filled | 42 | 0.966347 | **+0.000112** | 3 | 2.1e-05 | **PAYS** |
| 2026-08-26 | 002f | **`freq`** — transductive counts over 987,671 rows | 54 | 0.966977 | **+0.000609** | 1 | 4.1e-05 | **PAYS** |
| 2026-08-26 | 002g | **`impute`** — XGB imputation *alongside* the NaNs | 63 | 0.967170 | **+0.000193** | 3 | 2.2e-05 | **PAYS** |
| 2026-08-26 | 002h | **`te`** — nested stringified target encoding | 75 | **0.967910** | **+0.000740** | 1 | 2.4e-05 | **PAYS** |

**+0.003903 over raw**, and +0.003539 over the `xgb_baseline` feature block scored on the same
config (0.964371). Note the ranking is not the corpus ranking — see below.

**`null sd` is per-comparison, not a constant.** It is `sqrt(sigma_a^2/n_a + sigma_b^2/n_b)` from
the model-seed spread measured *on those two feature sets*; a block clears when |Δ| ≥ 2× it.
Single-seed rows use the larger neighbouring sigma. This replaces the flat 0.00011 gate, which
issue 003 measured on a different feature set — see the correction below.

#### sigma_model is a property of the feature set, not of the box

| feature set | cols | sigma_model |
|---|---:|---:|
| raw | 12 | 0.000005 |
| + budget | 21 | 0.000018 |
| + cat_isna | 24 | 0.000026 |
| + lattice | 36 | **0.000032** |
| + decimals | 42 | 0.000017 |
| + impute | 63 | 0.000017 |
| starter one-hot block (issue 003) | 34 | 0.000039 |

It spans **7×** across feature sets and is **not monotonic in column count** — it rises to a peak
near 36 columns and falls again once `freq`/`impute`/`te` supply dominant features. The reading:
seed sensitivity is highest when many columns are comparably useful and `colsample_bytree` has
real choices to make; it collapses when the model has either too little to choose from or one
obvious thing to lock onto. **Consequence: the 0.00011 gate from issue 003 is not portable.**
It belongs to the 34-column one-hot set it was measured on. Every ablation must carry its own
sigma, which is what the table above does.

#### The corpus ranking was wrong on this data

| block | corpus Δ | measured Δ | ratio |
|---|---:|---:|---:|
| `lattice` (decimal lattice) | +0.0001 | **+0.001609** | **16×** |
| `budget` (composition) | +0.0005 | +0.000626 | 1.3× |
| `freq` (transductive counts) | +0.0003 | +0.000609 | 2.0× |
| `decimals` | (part of lattice) | +0.000112 | — |
| `cat_isna` | +0.0001 | −0.000007 | **0×** |
| `impute` alongside | +0.0012 | +0.000193 | 0.16× |
| `te` (stringified TE + freq) | +0.0023 | +0.000740 | 0.32× |

Two separate effects, and they must not be confused:

1. **The decimal lattice was mis-ranked, badly.** It was listed fifth of six at +0.0001 and is the
   largest block by 2.6×. It is a value-encoding mechanism — `frac` and `d1` let the tree separate
   individual printed values — which is the same job `max_bin` and target encoding do.
2. **`te` and `impute` read low because they are substitutes, not because they failed.** By the
   time they are added, `lattice` and `freq` have already extracted most of the exact-value signal.
   Measured first instead of last they would price far higher. This is the same substitution the
   corpus found between `max_bin` and value encoding (~60% of the sum survives), and it is why the
   published deltas must never be added up.

#### `decimals` — a CV↑/LB↓ trap, defused rather than accepted

The first version of this block scored **+0.000133**, but `read_decimal_places` returns NaN
wherever the field is missing, and that NaN pattern is **bit-identical to the raw missingness
pattern**. So the block as written was a numeric missingness flag with a decimal bit attached —
and numeric missingness flags are excluded precisely because they raise CV and lower LB by
identifying the train/test split.

Three checks separated the halves:

| question | answer |
|---|---|
| Does the 1dp/2dp rate shift between train and test? | **No** — \|z\| < 1.7 in all six columns |
| Does the bit predict `y` on rows where the value is present? | **Yes** — P(y\|1dp) 0.6821 vs P(y\|2dp) 0.7121 on `daily_screen_time_hours`, gap +0.0300 |
| Is the NaN channel anything but missingness? | **No** — identical to the raw NaN mask |

Filling missing with the mode keeps the bit and discards the channel. The gain survives at
**+0.000112, i.e. 84% of the original** — so it was the generator's printed-precision artifact,
not the train/test shift. The block ships mode-filled.

#### Verdicts to carry into Phase 2

- Ship `raw + budget + lattice + decimals + freq + impute + te` — registered as `xgb_features`.
- **`cat_isna` is UNRESOLVED and is not shipped.** −0.000007 over three seeds per arm is not a
  loss either; the corpus's +0.0001 simply does not reproduce. Recorded, not booked.
- `max_bin` is still unpriced and must be measured *after* these blocks, not before: `lattice`,
  `freq` and `te` all do the same value-separation job, so Phase 0's +0.00142 for
  `max_bin` 255→2047 will shrink. That belongs to issue 006.

### What `xgb_baseline` established

| | |
|---|---|
| pooled OOF AUC | **0.964869** |
| per-fold | 0.96423 · 0.96483 · 0.96480 · 0.96585 · 0.96464 |
| fold mean ± sd | 0.964871 ± 0.000600 |
| **fold range** | **0.001629** |
| best iterations | 2802, 2521, 2641, 2818, 2764 (cap 3000) |
| cost | 8.5 min, ~100 s/fold, GPU |

Three things worth carrying forward:

1. **The fold range is 0.001629 — 30× the measured σ_delta of 0.000055.** Reading per-fold spread
   as uncertainty would make every gain below ~0.0016 look unmeasurable, when the real gate is
   0.00011. Issue 003 went further and showed *why* this member's range was so wide: on the frozen
   partition fold 3 is easier by +0.00097 and fold 0 harder by −0.00061, reproducibly. The
   0.001629 was mostly that fixed split effect, not model variance.
2. **Early stopping barely bound** — best iterations ran 2521–2818 against a 3000 cap, so the
   starter's `n_estimators` is close to limiting. Not tuned here on purpose (this member is a
   calibration point, ported verbatim), but any future XGBoost member should raise the cap.
3. **The published CV→LB line transfers.** Predicted offset +0.001629, observed **+0.001531** —
   residual −9.8e-05, inside the range a single public score cannot resolve. Until we have our own
   higher-CV points, the corpus line is usable as a pre-submission sanity check. Detail:
   [`../submissions/log.md`](../submissions/log.md).
4. **The OOF is mildly optimistic.** Early stopping selects the iteration count on the same
   validation fold the OOF is scored on. That is what the source notebook and the wider corpus do,
   so the number stays comparable to published ones — but it is not a clean nested estimate, and
   stack members whose weights get fitted must not inherit the habit.

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

Noise floor in that notebook: **0.00005** — within 9% of our own σ_delta, so the magnitudes
transfer. But that notebook gates at **1×** the floor, and our measurement says the honest gate is
**2σ = 0.00011**: two identical configs were seen 0.000101 apart.

**Everything in this table at or below +0.0001 is therefore unproven, not proven** — the decimal
lattice, the explicit `"__missing__"` level, 10 encoding folds, missingness augmentation,
constraint geometry, regime-aware stacking, and every "skip" verdict from −0.00001 to −0.00006.
Those are coin flips reported as findings. The verdicts worth acting on are the ones an order of
magnitude clear: stringified TE + frequency (+0.0023), imputed-alongside-NaN (+0.0012), ratios
(+0.0005), and on the negative side the DAE features (−0.00071), pairwise TE (−0.00040), and
pseudo-labelling (−0.0034). Issue 002 orders its blocks accordingly.

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
