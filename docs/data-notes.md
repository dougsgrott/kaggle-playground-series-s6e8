# S6E8 — verified dataset facts

Every claim here traces to a source: a mirrored competition page, the leaderboard snapshot, or a
specific cleaned competitor notebook under `analysis/nb_clean/`. Numbers without a source do not
belong in this file.

## Shape and target

| | | source |
|---|---|---|
| train | **691,369** rows × `id` + 12 features + `addicted_label` | `cdeotte__basic-eda-smartphone-addiction.py` |
| test | **296,302** rows | `hboyang__s6e8-rank-logit-regime-fusion-lb0-97125.py` (docstring) |
| positive rate | **0.709424** | verified locally, `scripts/check_env.py` |
| metric | ROC-AUC — only rank order matters | `wiki/competition/pages/evaluation.md` |
| public LB | **20% of test = 59,260 rows**; private = **237,042** | `raykkretzschmar__why-every-s6e8-notebook-above-0-97110-overfits.py` (`train_size=59_260`) |
| `sample_submission` | constant 0.7094 for every row | corpus EDA notebooks |

## Columns

```python
NUM_COLS = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "notifications_per_day",
            "app_opens_per_day", "weekend_screen_time"]           # float64, all nullable
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]     # object, all nullable
CAT_LEVELS = {"gender": ["Female", "Male", "Other"],
              "stress_level": ["High", "Low", "Medium"],
              "academic_work_impact": ["No", "Yes"]}
FRACTIONAL_COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
                   "work_study_hours", "sleep_hours", "weekend_screen_time"]
```

`age`, `notifications_per_day`, `app_opens_per_day` are whole numbers — they have no decimal part.

### Cardinalities (drive the `max_bin` choice)

Measured locally on `data/train.csv` by `scripts/check_env.py` (2026-08-26):

| column | distinct |
|---|---:|
| `weekend_screen_time` | **1,437** |
| `daily_screen_time_hours` | 1,389 |
| `social_media_hours` | 721 |
| `work_study_hours` | 600 |
| `sleep_hours` | 451 |
| `gaming_hours` | 401 |
| `notifications_per_day` | 231 |
| `app_opens_per_day` | 166 |
| `age` | 18 |

Rounded to one decimal the maximum distinct count collapses to 231.

**`max_bin` must be ≥ 1,437**, so 2047 is the right setting. Note the corpus attributes the
widest column to `daily_screen_time_hours`; locally it is `weekend_screen_time`. The two are the
only columns above 1,400 and together account for 91% of the `max_bin` gain, so the practical
conclusion is unchanged.

Source: local measurement; corroborated by `kitopl__max-bin.py`, `ern711__*` spline knot caps.

## Missingness — MCAR, but split-identifying

Every one of the 12 columns is nullable, **4–20% missing** (`age`/`gender` ≈ 4%,
`social_media_hours` ≈ 19%, screen-time columns 16–20%). Train counts:

```
age                       28929      daily_screen_time_hours   95854
social_media_hours       133995      gaming_hours             126821
work_study_hours          51518      sleep_hours               44480
notifications_per_day     67584      app_opens_per_day         80710
weekend_screen_time      112063      gender                    29034
stress_level              55148      academic_work_impact      44224
```

Two facts that only matter together:

1. **Missingness carries no target signal.** Standalone AUC of `n_missing` ≈ **0.502**; the target
   rate for missing vs present is within ~1 pp of 0.709 for every column.
2. **Train and test missing rates differ significantly in all twelve columns**, in mixed
   directions (8 columns missing more in test, 4 less; every gap ≥ 11 SE).

| column | train % | test % | diff (pp) | z |
|---|---:|---:|---:|---:|
| `social_media_hours` | 19.38 | 16.00 | −3.38 | −39.8 |
| `app_opens_per_day` | 11.67 | 8.68 | −3.00 | −44.1 |
| `daily_screen_time_hours` | 13.86 | 11.07 | −2.80 | −37.9 |
| `stress_level` | 7.98 | 6.62 | −1.35 | −23.3 |
| `gender` | 4.20 | 4.80 | +0.60 | +13.3 |
| `weekend_screen_time` | 16.21 | 17.11 | +0.90 | +11.1 |
| `sleep_hours` | 6.43 | 7.58 | +1.14 | +20.7 |
| `age` | 4.18 | 5.78 | +1.60 | +34.6 |
| `gaming_hours` | 18.34 | 20.05 | +1.71 | +19.9 |
| `notifications_per_day` | 9.78 | 11.55 | +1.77 | +26.6 |
| `work_study_hours` | 7.45 | 9.37 | +1.92 | +32.2 |
| `academic_work_impact` | 6.40 | 8.68 | +2.28 | +40.6 |

**Consequence: `is_missing` flags separate train from test while predicting nothing.** Expect
CV↑ / LB↓. Measured at −0.00001 to 0 in several notebooks.

**Adversarial validation** (train vs test, 5-fold, pooled 987,671 rows):

| variant | adversarial AUC |
|---|---:|
| raw features, NaN kept, no flags | 0.564146 |
| raw + 12 `is_missing` flags | 0.564958 |
| `is_missing` flags only | 0.565120 |
| NaNs imputed away | **0.503339** |
| complete rows only | **0.498486** (95% CI [0.496194, 0.500552]) |

"The missingness is not part of the shift; it is the whole shift." Once NaNs are removed, train
and test are statistically indistinguishable — **there is no covariate drift to correct for**.

One refinement worth keeping: the 9 numeric `__isna` flags carry exactly zero gain (LightGBM's
native NaN routing already produces that split), but the **three categorical `__isna` flags do**,
with `academic_work_impact__isna` the largest.

A model restricted to fully complete rows scores **0.9708** vs **0.9643** on all rows — missingness,
not modelling, is what caps the achievable score.

Sources: `dariushafshar__s6e8-what-actually-helps.py`,
`dariushafshar__5-of-9-numeric-missingness-flags-lose-to-noise.py`,
`abdullahsafwan333__s6e8-complete-eda-insights-sap.py`, discussion topics 731764, 732427, 733214.

## The generator

### The source label rule

The original dataset (`Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv`, 7,500 rows, 70.77%
positive) is a two-rule lookup with `dsth = daily_screen_time_hours`, `smh = social_media_hours`:

```
p = 1    if dsth > 8  or  smh > 4
p = 0    if dsth <= 6 and smh <= 4
p = 0.5  otherwise                       # 1,025 rows, actual 0.4556, XGB AUC 0.510 ± 0.033
```

Bayes-optimal AUC of that rule on the original: **0.9888**.

### What the generator did to it

Running the identical rules on the 691k synthetic rows:

- the "always-zero" region is now **33% addicted**, the "always-one" region picks up counterexamples;
- the two rules together fall from **0.9888 → 0.835**;
- the previously-random middle band is **a third of all synthetic rows**, and a GBDT trained on it
  alone goes from **0.50 (original) → 0.896 (synthetic)**.

**Do not hard-code the rules — they cap at 0.835. The generator is the signal.** But the boundary
region is where all residual error lives, which is why it is the most promising research target.

Source: discussion topics 732428, 732434.

### The budget identity — exact, and exploitable

`daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours` holds in
**100.00000%** of all 859,029 train+test rows, minimum gap exactly 0.000, 546 rows on the boundary.
The same constraint is violated in **26–61%** of the real source rows — the generator *learned and
enforced* it.

Therefore `other_screen = daily − (social + gaming + work)` is a real reconstructed latent
variable and axis-aligned trees cannot construct a 4-term linear combination from splits.
`weekend_screen_time` has no such constraint (useful control).

**Verified locally** (`scripts/check_env.py`, 2026-08-26): **zero violations** on the
**421,427** train rows where all four columns are present, and `other_screen` standalone
**AUC = 0.7649** — the corpus figure of 0.765 reproduces exactly.

Watch the arithmetic: `df[PARTS].sum(axis=1)` skips NaNs by default, which silently weakens the
check to "`daily` is present" and inflates the row count to 595,515. Use
`sum(axis=1, min_count=3)`.

Other repaired constraints (generated → original violation rate): `social + gaming > screen_time`
0% → 26.3%; `social > screen_time` 0% → 9.1%; `work_study > screen_time` 0% → 9.1%.

Source: `tamerlanomralinov__s6e8-lookup-transformer-insights-lb-0-97041.py`, discussion 734501.

### What the generator did *not* learn

In the source data `weekend_screen_time` = weekday screen time × a factor confined to
**[1.044, 1.965]**. The generator did not reproduce that: **26% of competition rows fall outside
the envelope, in train and test alike** — so it is not exploitable as leakage. Inside the plausible
band the target fires 81%, past 2.5× it fires 25%, base rate 71%. It is a **hump**, so no monotone
model (logistic regression, rank-average blend, monotone-constrained booster) can represent it.
Handing the ratio to LightGBM measures at **−0.00007** against a seed spread of ±0.00004 — a null.

Source: discussion 733983.

### Values are lookup keys, not just quantities

- Columns are quantised to 2 decimals; train and test share nearly all distinct values.
- `notifications_per_day` has **univariate AUC 0.492** — no monotone signal at all — yet its
  per-value residuals correlate **0.72** across two independent halves. Per-value target-rate
  sd = **0.1914** vs 0.0087 expected from sampling (22×); the mean absolute rate difference
  between *adjacent* integer values is **0.2248**.
- `notifications_per_day` + `app_opens_per_day` alone reach **AUC 0.83**; dropping them costs
  ~0.019.
- The first decimal digit of `daily_screen_time_hours` shows an **8.5 pp swing** in addiction rate
  across the ten digits, on 50–68k rows per digit. The effect is concentrated in that one column —
  the circulating "1-decimal rows run lower across the board" claim is wrong.
- **But identity encoding is the wrong way to use this.** Casting `gaming_hours` /
  `work_study_hours` to unordered categories scores *below* omitting them; the second decimal
  carries nothing; the **continuous fractional part `v − floor(v)`** beats the whole digit block.
  What identity encoding forfeits is ordering, not statistics.

Sources: `tomasa2__s6e8-what-moved-the-score-and-what-didn-t.py` §8.1,
`dariushafshar__exact-values-fail-as-lookup-keys.py`, `dariushafshar__s6e8-what-actually-helps.py`,
discussion 737422.

## Measurement discipline

### The noise floor — measured locally

Measured on this box 2026-08-26: **55 XGBoost-GPU fits, 6 partition seeds x 6 model seeds** on the
starter feature block (pooled AUC ~0.9644, the same regime as the members). Every cell is in
[`noise_floor.json`](noise_floor.json); the method is `scripts/noise_floor.py`.

"The noise floor" names two different quantities, and using the wrong one is how a nonexistent
gain gets shipped:

| quantity | value | what it gates |
|---|---:|---|
| **σ_partition** | **0.0000193** | comparing our CV to a number computed on a *different* split |
| σ_model | 0.0000391 | one config re-seeded on the frozen partition |
| **σ_delta** = √2·σ_model | **0.0000552** | an **A-vs-B delta** on the frozen partition — i.e. every row in [`experiments.md`](experiments.md) |
| max observed null delta | 0.000101 | the largest gap two *identical* configs actually produced |
| mean per-fold range | 0.001215 | **nothing.** 22× σ_delta |

**σ_delta is the operative one.** No experiment in this repo varies the partition — they all hold
it at seed 42 and change only the config — so the null distribution a delta must beat is the
spread of two re-seeded arms of the *same* config, not the spread across splits.
Practical gate: **an ablation must clear 2σ_delta = 0.00011.**

Three findings worth carrying:

1. **The model seed matters twice as much as the partition seed** (0.0000391 vs 0.0000193). Pooling
   691,369 rows makes the choice of split nearly irrelevant — every row is predicted exactly once
   by a model trained on ~553k rows either way — while `subsample`/`colsample_bytree` redraw in all
   five models. So *repeated CV across partitions is the wrong way to buy precision on an
   ablation.* Averaging model seeds is the right way: σ_delta falls as √n, to 0.000032 at 3 seeds
   per arm and 0.000025 at 5.
2. **The per-fold range is a property of the split, not an error bar.** On the frozen partition
   fold 3 is genuinely easier by **+0.00097** and fold 0 harder by **−0.00061**, and that pattern
   reproduced across all six model seeds. It is why the frozen partition's fold range
   (0.00142–0.00170) is disjoint from every other partition's (0.00050–0.00116). Single-fold
   probes on fold 0 therefore read roughly **0.0006 low** against a pooled number.
3. **The corpus constant was right by luck.** Its 0.00005 is within 9% of our σ_delta (0.0000552),
   but it was *labelled* as the partition quantity — and our partition quantity is 0.0000193, 2.9×
   smaller. The number survived; the attribution did not.

### LB resolvability

| comparison | smallest distinguishable difference |
|---|---|
| two highly correlated submissions | **~7e-05** |
| two weakly correlated submissions | ~1.3e-03 to 1.8e-03 |

Paired reseed sigma on the 59,260-row public split is **0.00009–0.00011**, so ±1 sigma spans
~60 teams between rank 10 and rank 100. Resolvability is not a fixed threshold — it scales with
1 − ρ.

Price of a rank near the plateau: **~1.6 ranks per 1e-06 of AUC**. And because Kaggle keeps your
best public score, resubmitting an identical-quality file has **P(rank improves) ≈ 0.486** for free.
Roughly half of any apparent gain from resubmission is best-of-N, not skill.

### The CV→LB offset is a line, not a constant

Measured over 14 submissions: the offset **decays from +0.00150 at CV 0.9660 to +0.00109 at
CV 0.9696**, with `corr(CV, offset) = −0.99`. Mechanism: test predictions average 5 fold-models
while OOF comes from 1, and a large stack has already averaged that variance away.

**Fit a line. Do not pin a constant.**

Sources: `tomasa2__s6e8-what-moved-the-score-and-what-didn-t.py` §9, discussion topics 733214,
733618, 734005.

## Private-LB risk

- Public LB is 59,260 rows; private is 237,042 (4×).
- Public-LB weight selection **reverses on held-out labels**: the honest OOF optimum said add 12%
  of a student model, the public LB rewarded subtracting 8%.
- **Three of the seven finished Season-6 boards erased their entire public top ten.** In S6E2,
  S6E6 and S6E7 *zero* public-top-10 teams stayed in the private top 10; the S6E7 public winner
  finished private rank 440. The kept-count distribution is bimodal — never between one and four.
- No early-warning signal (episode metric, fork-cluster density, top-ten packing) separates the
  wiped boards from the held ones. What is left is base rates.

Sources: `raykkretzschmar__why-every-s6e8-notebook-above-0-97110-overfits.py`,
`georgymamarin__s6e8-will-your-0-971-survive-the-private-split.py`,
`georgymamarin__three-of-seven-s6-boards-erased-the-public-top-ten.py`.

## The fold contract

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train.addicted_label)
# over train.csv in ORIGINAL FILE ROW ORDER
# OOF/test arrays are POSITIONAL, no id column:
#   oof/oof_<name>.npy   shape (691369,)  train.csv order
#   oof/test_<name>.npy  shape (296302,)  test.csv order
```

Materialised as `data/folds.npy` by `scripts/make_folds.py`. This is the convention the entire
public OOF ecosystem uses, so any borrowed member drops straight in — and any member built on a
different partition does not.

`id` carries no drift: AUC 0.5007, and the target rate is flat across 20 id chunks.

## Leaderboard shape (snapshot 2026-08-26 14:22 UTC, 2,987 teams)

| rank | score |
|---:|---:|
| 1 (Chris Deotte) | 0.97184 |
| 2 | 0.97154 |
| 10 | 0.97132 |
| 25 | 0.97129 |
| 50 | 0.97128 |
| 100 | 0.97127 |
| 200 | 0.97117 |
| 500 | 0.97077 |
| 1000 | 0.96743 |

Ranks 17–100 span 2e-05 — well inside the noise floor. That plateau is **largely one shared file**:
several top notebooks read a public `submission.csv` and re-emit it. Rank 1 is **~3 sigma clear**
of rank 50, so that gap is real and unpublished.

Source: `wiki/leaderboard/snapshots/playground-series-s6e8-publicleaderboard-2026-08-26T14:22:15.csv`,
`abhirajhiwale__s6e8-mapping-the-public-plateau-0-97128.py`,
`amanatar__s6e8-elite-rank-average-ensemble-0-97123.py`.
