# S6E8 — Roadmap

**Playground Series S6E8 — Predicting Smartphone Addiction.** Binary target `addicted_label`,
ROC-AUC, deadline **2026-08-31 23:59 UTC**. Written 2026-08-26 → **5 days, ~50 submissions
(10/day), 2 final selections**.

Dataset facts referenced throughout: [`data-notes.md`](data-notes.md).
Append-only ledgers: [`experiments.md`](experiments.md), [`../submissions/log.md`](../submissions/log.md).

---

## 1. Where this starts

The repo has a complete `kaggle-wiki-kit` knowledge base — the competition mirror, 49 discussion
threads, **582 cleaned competitor notebooks** in `analysis/nb_clean/`, and a leaderboard snapshot —
and **no solution code, no data, and no submission**. Kaggle handle `dougsgrott` is not on the
board.

That is a favourable position, not a bad one: the corpus contains measured ablations of ~70 ideas,
working code for every technique that pays, and the exact fold convention the whole community
shares. Competition rules 2.6 and 3.6.b make all of it reusable. The work is porting and
recombining, not discovering.

## 2. Objective and strategy (decided)

- **Objective**: best expected **private**-LB finish. Playground awards no medals; the public board
  is not the target.
- **Strategy**: **own models first.** Every stack member is trained locally on the frozen folds.
  Public OOF *arrays* are downloaded once, late, and used only as (a) a correctness benchmark for
  our OOF contract and (b) the second of the two final submissions.
- **Research track included**, timeboxed.

### Why this shape

Three findings from the corpus set the strategy:

1. **The public plateau is one shared file.** Ranks 17–100 sit within 2e-05 of each other because
   several top notebooks read a public `submission.csv` and re-emit it. Joining that cluster costs
   nothing and buys nothing on the private split.
2. **Selecting on public score actively backfires.** Zero public-top-10 teams stayed in the private
   top 10 in S6E2, S6E6 and S6E7; the S6E7 public winner finished private rank 440.
3. **The published pyramid buys almost nothing above a good honest stack.** Best single 0.968–0.970
   → Layer-2 linear stack 0.9707–0.9710 → regime fusion 0.97125 → CSV rank-blending 0.97128. Layers
   3–4 together are worth ~0.00003–0.00018, entirely inside the public standard error.

So the plan spends its effort where the returns are: a strong, honest Layer-0/Layer-2 stack of our
own members, selected on nested CV.

## 3. Expected outcome

| milestone | OOF | expected LB |
|---|---|---|
| Phase 1 baseline (ported XGB starter) | **0.964869 actual** | **0.96640 actual** |
| Single tuned member with value encodings | ~0.9686 | ~0.9700 |
| Phase 3 honest stack of ~20 own members | **0.9690–0.9705** | **0.9702–0.9716** |
| Phase 4 plateau-consensus pick (pick B) | — | ~0.9712 |

The CV→LB offset is **a line, not a constant** — it decays from +0.00150 at CV 0.9660 to +0.00109
at CV 0.9696 (`corr = −0.99`). Fit it from our own submissions; do not pin a value.

**First point measured (issue 004):** CV 0.964869 → LB 0.96640, offset **+0.001531** against a
predicted +0.001629 — residual −9.8e-05. The published line transfers, so it is usable as a
pre-submission sanity check until we have higher-CV points of our own. Next points wanted at
CV ≈ 0.967 and ≈ 0.970, spread across days: the line's value is its slope, and clustered points
cannot measure one. Detail: [`../submissions/log.md`](../submissions/log.md).

---

## Phase 0 — Ground truth *(today, ~1 h)*

**Done** — 2026-08-26. Detail: [`issues/001-phase-0-ground-truth.md`](issues/001-phase-0-ground-truth.md).

1. ML dependencies in `pyproject.toml`; `uv sync`. Installed: pandas 3.0.5, numpy 2.5.2,
   lightgbm 4.7.0, xgboost 2.1.4 (pinned <3.0), catboost 1.2.10, torch 2.6.0+cu124.
2. Competition data in `data/`; shapes verified **(691369, 14)** and **(296302, 13)**, positive
   rate **0.709424**.
3. Repo skeleton: `docs/`, `src/s6e8/`, `scripts/`, `data/`, `oof/`, `submissions/`. The package
   is importable as `s6e8` (`[tool.uv.build-backend] module-name`).
4. **Folds frozen** — `data/folds.npy`, sizes `[138274, 138274, 138274, 138274, 138273]`,
   sha256 `9571f18b…bee824`. The digest is the contract: a member built on any other partition
   cannot enter a stack, and this is what proves it.
5. `scripts/check_env.py` → **PASS, 16 checks**. GPU: RTX 4050 Laptop, 6.0 GB. The gate also
   guards the three environment traps found today — see
   [`issues/001`](issues/001-phase-0-ground-truth.md): XGBoost 3.x silently training on CPU
   (CUDA 13 wheel vs a 12.9 driver — `xgboost` is pinned `<3.0`), OpenMP thread
   oversubscription (9× slower at 4 threads than 1 on this loaded box), and the shared
   `nvidia/nccl/lib` directory that breaks `import torch` when either NCCL package is removed.
6. `scripts/benchmark.py` → measured per-member cost, below.

### Three things the gate settled

- **The budget identity holds exactly**: zero violations on the 421,427 rows where all four
  columns are present, `other_screen` standalone **AUC 0.7649** — the corpus figure reproduces.
  Note `df[PARTS].sum(axis=1)` skips NaNs and silently weakens the check to "`daily` is present";
  `min_count=3` is required.
- **pandas 3.0.5 is installed, so the target-encoding trap below is live, not hypothetical.** The
  gate measures it: naive `astype(str)` covers 2 of 3 probe rows. Use
  `src/s6e8/data.py::string_levels` and nothing else to build encoder keys.
- **The widest column is `weekend_screen_time` (1,437 distinct), not `daily_screen_time_hours`
  (1,389)** — a correction to the corpus. Both are above 1,400, so `max_bin = 2047` stands.

**Exit criterion**: met.

---

## Phase 1 — Feature module and honest harness *(day 1)*

The corpus already tells us what pays. Build **only** the measured-positive set, and record the
measured reason for every exclusion so nobody re-litigates it.

### `src/s6e8/features.py` — build these

| feature block | measured delta | source |
|---|---:|---|
| **Stringified target + frequency encoding, all 12 columns, smoothing 10, nested 5×5** | **+0.0023 CV / +0.0017 LB** | `tomasa2__*` §6 |
| **Imputed columns *alongside* the raw NaN columns** (never replacing) | **+0.0012** | `tomasa2__*` §5 |
| **CatBoost's own ordered target statistics on raw string levels** | **+0.0004** over the hand-rolled encoder | `tomasa2__*` §6.1 |
| **Transductive frequency counts over train+test (987,671 rows)** | **+0.00032 solo** | `tomasa2__*` §8.3, `najiama__single-lgbm-*` |
| `other_screen` residual + composition/ratio block | +0.0005 | `cdeotte__simple-xgb-starter.py` |
| **Decimal lattice: `frac_c = v − floor(v)`, `d1_c = floor(v*10) % 10`** | +0.0001 | `tomasa2__*` §7 |
| Decimal-place counts read from **raw CSV text** | part of the "structural artifacts" +0.00088 block | `dariushafshar__s6e8-what-actually-helps.py` |
| 3 categorical `__isna` flags only | small but real | discussion 733214 |

### Do not build these — each one was measured negative

`pseudo-labelling confident test rows` (**−0.0034**, the single worst idea in the corpus) ·
`denoising-autoencoder features` (−0.00071) · `pairwise target encoding` (−0.00040) ·
`multi-resolution TE` (−0.00033) · `monotone constraints` (−0.0003) · `TE smoothing 50/200`
(−0.00006 / −0.00030) · `tree depth 9–13` (to −0.0011) · `concatenating the original 7,500-row
dataset` (−0.0001) · `numeric is_missing flags` (−0.00001, and they identify the split) ·
`identity/digit categoricals` (below baseline — the continuous fractional part wins) ·
`behavioural ratios, sleep_deficit, weekend-ratio, total_weekly_screen_time` (all lose).

### Two traps to code around

**The pandas ≥ 3.0 target-encoding trap — silent, no error, no warning.** On pandas 2.x,
`astype(str)` writes the literal `"nan"` so missing values become their own level. On pandas 3.0
the new `str` dtype *preserves* NA, so `groupby` silently drops every missing row from the level
statistics and `.fillna(global_mean)` quietly hands them the base rate — measured coverage 595,515
of 691,369 rows. Always:

```python
df[c].astype(object).fillna("__missing__").astype(str)
assert groupby(...).size().sum() == len(df)
```

**`max_bin` and value encoding are substitutes, not additive.** Raising `max_bin` 255 → 2047 is
worth +0.0024 on raw columns but only +0.0005 once frequency encoding is present; measured
separately at +0.0024 and +0.0018, doing both yields +0.0022 — just under 60% of the sum. Set
`max_bin` ≥ the largest distinct-value count (1,437 raw → 1023–2047) and then re-measure the
encodings, never add the two published gains together.

### Harness

- `src/s6e8/cv.py` — loads `data/folds.npy`, runs a member, writes the positional OOF contract.
- **Establish the noise floor first**: repeated 5-fold across 3 partition seeds, std of the *means*
  ≈ **0.00005**. The per-fold range inside one run is ~10× larger and is not the noise floor.
- Port `analysis/nb_clean/cdeotte__simple-xgb-starter.py` → first submission, first CV↔LB anchor
  point in `submissions/log.md`.

**Exit criterion**: a feature module with an ablation table in `docs/experiments.md` where every
row is expressed in multiples of the measured noise floor, and one submission on the board.

---

## Phase 2 — Own member zoo *(days 1–3 — the main effort)*

~20 members, all on `data/folds.npy`, each exporting `oof/oof_<name>.npy` + `oof/test_<name>.npy`.

The corpus contains **ready-made, fold-aligned suites whose source we already have** — porting and
running them locally is the fastest route to a diverse zoo that is genuinely ours:

- `analysis/nb_clean/abdullahsafwan333__s6e8-{lightgbm,xgboost,catboost,realmlp,tabm,ft-transformer}-sap.py`
  — six members, identical FE, one per family, each already writing OOF artifacts.
- `analysis/nb_clean/omidbaghchehsaraei__*` — fifteen members covering XGBoost, CatBoost, RealMLP,
  TabM, ResNet, TabNet, FT-Transformer, TabTransformer, CNN, FastAI, FLAML, Lookup-Transformer.
- `analysis/nb_clean/beicicc__s6e8-fold-safe-{lattice-target-encoding,realmlp,tabnet}.py` —
  provenance-verified with manifests and SHA-256 hashes.

### Priority order (highest value first)

1. **CatBoost with all 12 raw string levels as `cat_features`** — its native ordered target
   statistics beat a hand-rolled encoder by +0.0004 for one argument.
   `iterations=6000, learning_rate=0.03, depth=6, eval_metric="AUC", task_type="GPU"`.
2. **LightGBM + exact-value TE/freq, `max_bin` 1023–2047, `num_leaves` 127+.** The best documented
   single LGBM is CV 0.96862 → **LB 0.96990** (`najiama__single-lgbm-model-lb-0-96990-cv-0-96862.py`).
3. **XGBoost `device="cuda"`** across feature views — cheapest member to produce (~2–6 min for a
   full 5-fold), so it is the right vehicle for view diversity.
4. **Lookup-Transformer** — *the load-bearing member of every public stack*. A leave-one-out test
   found dropping it costs **−0.000106, 6.6× the next member**, while 73 of 74 other members cost
   under 0.00002. Per-column exact-value embedding table (0 reserved for NaN) + an added
   periodic-linear branch + 6 derived budget tokens + a 4-layer pre-norm transformer.
   Source: `analysis/nb_clean/tamerlanomralinov__s6e8-lookup-transformer-insights-lb-0-97041.py`.
5. **RealMLP** — best published *single-model* LB in the corpus (**0.97009 / 0.97014**), and a
   self-contained PyTorch implementation exists so no `pytabkit` dependency is required.
   Source: `analysis/nb_clean/zhenruiweng__s6e8-public-lb-0-97009-single-model-realmlp.py`.
6. **TabM** — `arch_type="tabm-mini-normal", tabm_k=24, num_emb_type="pwl"`; this is the notebook
   credited with originating the stringified-TE idea.
7. **Spline transformer** — the only *new* architecture that measurably paid (solo OOF 0.96680,
   mean correlation 0.930, +0.00004 paired stack delta). `analysis/nb_clean/ern711__*`.
8. **One deliberately weak, structurally different member** — RandomForest at ~0.943. Measured to
   earn stack weight where a stronger-but-similar HistGB earns none.

### Admission rule for a new member

**Decorrelated *and* comparably strong.** Rank correlation above ~0.99 to an existing member, or
more than ~0.006 AUC weaker, and it earns weight 0.000 — measured repeatedly:

| pair | rank corr | blend contribution |
|---|---:|---|
| LightGBM → XGBoost, same features | 0.9972 | +0.00003 |
| same algorithm, different Optuna params | 0.99+ | weight **0.000** |
| same NN, different seed | — | +0.00002 |
| GBDT → NN with value embeddings | 0.974 | weight **0.22** |
| GBDT → ExtraTrees | 0.967 | weight **0.000** (0.0064 too weak) |

There is a visible strength cliff around solo OOF 0.966: contribution tracks solo OOF more than it
tracks decorrelation.

**Also**: train the NN members with **missingness augmentation** (`mask_prob ≈ 0.3`, training only,
reusing the real missing-value pathway rather than dropout) — +0.0014 solo. Expect the *ensemble*
gain to be smaller, because a missing-robust NN reasons more like the tree models.

**Compute budget — measured, not estimated.** One fold (553,095 train / 138,274 valid), raw 12
columns, `scripts/benchmark.py` on 2026-08-26 at load average ~45:

| member | config | rounds | 1 fold | fold AUC | peak RSS |
|---|---|---:|---:|---:|---:|
| XGBoost **GPU** | depth 7 | 400 | **27.2 s** | 0.95894 | 554 MB |
| LightGBM CPU (2 threads) | leaves 63, `max_bin` 255 | 200 | 85.1 s | 0.95852 | 430 MB |
| LightGBM CPU (2 threads) | leaves 255, `max_bin` 255 | 200 | 94.7 s | 0.96161 | 430 MB |
| LightGBM CPU (2 threads) | leaves 255, **`max_bin` 2047** | 200 | 139.5 s | **0.96303** | 430 MB |
| CatBoost **GPU** (native cats) | depth 6 | 200 | 102.8 s | 0.94124 | 1,076 MB |
| CatBoost CPU (4 threads) | depth 6 | 200 | 204.3 s | 0.94097 | 1,076 MB |

Scaled to a production 5-fold run at 3,000 rounds:

| member | 5 folds |
|---|---:|
| XGBoost GPU | **~17 min** |
| CatBoost GPU | ~2.1 h |
| LightGBM `max_bin=2047` | ~2.9 h |

**This is 5–10× slower than a quiet machine would give**, because the box is running at load
average ~45 on 20 cores. Consequences for Phase 2:

- **XGBoost GPU is the workhorse.** It is 4–8× cheaper per member than anything else and is
  unaffected by the CPU contention. Use it for feature-view diversity.
- **Budget LightGBM and CatBoost members deliberately** — two or three each, not ten.
- Re-run `scripts/benchmark.py` when the machine is quiet; these numbers are a load-dependent
  floor, not a constant.

Memory is *not* the constraint for single members: peak RSS never exceeded 1.1 GB. The fusion
layer is where RAM bites — see the risk register.

**Exit criterion**: ≥ 15 members in `oof/`, each with a solo OOF AUC and a rank-correlation matrix
recorded in `docs/experiments.md`.

---

## Phase 3 — The honest stack *(days 3–4)*

**Logit-space L2 logistic regression, nested. Not hill climbing, not plain averaging.**

- **Stack on logits, not probabilities** — +0.00047 in a heterogeneous library. The target
  saturates hard; the top screen-time decile has rate 1.000 where probabilities have no resolution.
- **Logistic regression beats greedy hill climbing** by +0.000102 (SE 0.000011, 5/5 folds) on
  identical members and folds. Hill climbing can only *add*; a stacker can subtract, and
  weak-but-decorrelated members earn negative coefficients as corrections.
- **Plain averaging of heterogeneous models lands ~0.006 *below* the best single model.** Learned
  weights land above it.
- **`C` is member-count dependent** — the argmax moves (1.0 at 56 members, 0.1 at 66, 0.03 at 74).
  Search `[0.03, 0.1, 1.0]` and confirm the plateau. Use `tol=1e-5`; `1e-6` gives identical OOF to
  6 decimals and takes 880 s instead of 500 s.
- **Assert convergence.** `assert max(n_iter_) < max_iter` — a non-converged stack reads *high*.

Reference implementation to port: `analysis/nb_clean/adarsh1077__s6e8-diversity-beats-strength.py`
(the cleanest, most auditable stacker in the corpus, with member QC gates and a leave-one-out
ablation) and `analysis/nb_clean/szymonkapiski__s6e8-honest-oof-blend.py`.

**Member QC gates worth copying verbatim**: drop on non-finite values; drop AUC < 0.90; drop
`ks_2samp(rank(oof), rank(test)) > 0.05`; drop exact duplicates by md5 of the OOF bytes.

If time allows, add the **regime arm** — every member's logit interacted with `complete` (0 NaNs),
`severe` (≥ 4 NaNs) and a normalised per-row disagreement, rank-mixed into the plain arm at weight
**1/3**. Worth +0.000028 (5/5 folds): worse alone, better mixed, because the two arms model
different things — what the pack thinks, versus when to distrust the pack. Normalise the test
design with **train** statistics; test spread is systematically smaller because test predictions
are 5-fold averages.

**Memory.** The published 205-member fusion builds a ~1,205-column float64 matrix ≈ **6.7 GB** and
will not fit in 7 GB of system RAM. Prune to ~60–100 members, work in float32, memmap the pool.

**Exit criterion**: a nested-CV OOF number for the stack, with the naive-vs-nested premium
reported so the optimism is visible rather than assumed.

---

## Phase 4 — Benchmark and the second pick *(day 4)*

Download the public OOF libraries **once**, and use them for exactly two things.

| dataset | owner | members |
|---|---|---|
| `s6e8-oof-library-47-models` | szymonkapiski | 47→74 |
| `s6e8-oof-prediction-library` | boltuzamaki | ~40 |
| `s6e8-golem-oof-library` | dariushafshar | 7 |
| `s6e8-adarsh-oof-library` | adarsh1077 | 5 |
| `s6e8-fm-lattice-blend-members` | raykkretzschmar | 5 |
| `s6e8-50-weakest-oof-models` | szymonkapiski | 50 |
| `predicting-smartphone-addiction-oof-submission-csv` | najiama | blends 07–19 |

1. **Correctness benchmark** — our OOF contract must reproduce their manifest AUCs and align
   positionally. A mismatch means our fold file is wrong, and that is worth knowing before Phase 5.
2. **Final pick B** — the plateau-consensus submission.

**The level-2 trap — mandatory filter.** Adding one member to an 83-model stack scored nested CV
+0.0000505 (t 7.39, 5/5 folds) and moved the public LB **down 0.00031** — wrong sign, six times the
size. The member was somebody else's level-2 stack: its OOF is row-honest, but its *features* are
level-1 OOF columns built on a different partition, so information about row *i* reaches the fit
through other rows' columns. On test none of that exists. It passes every isolated honesty check.
**Filter any external manifest's `family` field on `stack|blend|ensemb|meta`.**

Also note a loader bug worth avoiding: replace `"oof" → "test"` in the **filename only**. The
containing folder is often called `s6e8-oof-prediction-library`, and a whole-path replace silently
drops 47 members.

**Exit criterion**: our fold file verified against an external library; pick B written and logged.

---

## Phase 5 — Selection *(day 5)*

Two finals, by design, chosen before the last day so the choice is not made under time pressure:

- **Pick A — our honest nested-CV stack.** The bet that a real CV edge survives a 237,042-row
  private split.
- **Pick B — the plateau consensus.** The bet that the herd is right.

Neither is selected on public score. Write the rationale into `docs/experiments.md` *before*
seeing the final public standing.

Submission budget: 10/day. Spend them on **measuring the CV→LB line** (a few well-separated CV
levels), not on climbing. A change that moves 40 places but moves the score by less than 1e-4 has
measured nothing.

---

## Research track *(timeboxed ~half a day, day 2 or 3)*

Rank 1 (0.97184) is **~3 sigma clear** of rank 50 — real, and unpublished. Chris Deotte's four
public notebooks are clean starters with no special tricks, so the edge is elsewhere. Leads, in
order of expected value:

1. **Model the smoothed rule boundary explicitly.** The generator smeared a hard two-rule lookup
   (`dsth > 8 or smh > 4` → 1; `dsth ≤ 6 and smh ≤ 4` → 0). All residual error concentrates in the
   5–8.7 h screen-time band. Build signed distance-to-threshold features and a band indicator, and
   check whether a band-specialised model earns stack weight. Precedent: the Optuna fusion notebook
   already applies rank-preserving local re-sorts inside the 3–6 h and 6–7.8 h bands.
2. **Impute the two rule drivers transductively.** `daily_screen_time_hours` is missing on ~14% of
   rows and `social_media_hours` on ~19% — exactly the rows where the boundary is unrecoverable.
   Imputing *alongside* the raw NaN columns is already known to be worth +0.0012.
3. **Per-regime specialisation.** Complete rows score 0.9708 vs 0.9643 overall. A model trained
   only on complete rows, entered as a stack member with a missingness-regime interaction, is
   cheap to test.
4. **The unanswered forum question.** `gender`, `stress_level` and `academic_work_impact` have ~0
   *global* mutual information with the target, yet all remaining error lives in the 5–8.7 h band.
   Nobody has published whether they carry *conditional* signal inside it.

Each lead gets a `docs/issues/NNN-slug.md` and a row in `docs/experiments.md` whether it works or
not. Timebox is a hard stop — the known-good pipeline has priority.

---

## Risk register

| risk | likelihood | mitigation |
|---|---|---|
| **Private shakeup** — public plateau reorders essentially at random | high | Two structurally different final picks; select on nested CV; never on public score |
| **RAM — worse than nominal** | high | The box reports 7.8 GB but measured **~2 GB actually available** during Phase 0 (unrelated `pytest` processes held ~2 GB, Claude processes ~1.5 GB, and 1.3 GB of the 2 GB swap was already in use). Load only needed columns as float32; never one-hot the full frame; prune the fusion pool to ~60–100 members and memmap it. The published 1,205-column float64 matrix (~6.7 GB) is out of reach by a wide margin. Check `MemAvailable` before launching a long run. |
| **5-day clock** | high | Phase 2 is the priority; Phases 3–4 have working reference implementations to port; the research track is timeboxed |
| **Silent pandas ≥ 3.0 TE trap** | medium | `astype(object).fillna("__missing__").astype(str)` + a coverage assertion in the encoder |
| **Level-2 leak** from a borrowed member | medium | `family` filter; CV↑/LB↓ is the signature |
| **Fold misalignment** in a borrowed array | medium | Phase 4 benchmark reproduces external manifest AUCs before anything is trusted |
| **Chasing noise** | medium | Noise floor measured in Phase 1; every ledger row expressed in multiples of it |

---

## Task index

Execution order is **not** numbering order — see [`issues/README.md`](issues/README.md) for why.
Next up: **003 → 002**, with 014 running in parallel from day 2. (004 is done.)

| # | phase | task | status |
|---|---|---|---|
| 001 | 0 | [Environment, data, frozen folds](issues/001-phase-0-ground-truth.md) | **done** |
| 004 | 1 | [Baseline member + first submission](issues/004-baseline-first-submission.md) | **done** — OOF 0.964869 → LB 0.96640 |
| 003 | 1 | [Noise floor, local measurement](issues/003-noise-floor.md) | **next** |
| 002 | 1 | [`features.py` — measured-positive blocks](issues/002-feature-module.md) | open |
| 007 | 2 | XGBoost GPU members across feature views — **the workhorse** | open |
| 005 | 2 | CatBoost native-categorical member (blocked by [016](issues/016-catboost-gpu-eval-metric.md)) | open |
| 006 | 2 | LightGBM value-encoding member (`max_bin` sweep) | open |
| 008 | 2 | Lookup-Transformer member | open |
| 009 | 2 | RealMLP + TabM members, missingness augmentation | open |
| 010 | 3 | Nested logit-space logistic stack + QC gates | open |
| 011 | 3 | Regime arm, rank-mixed at 1/3 | open |
| 012 | 4 | Public OOF benchmark + pick B | open |
| 013 | 5 | Final selection and rationale | open |
| 014 | R | [Boundary-band research track](issues/014-boundary-band-research.md) — day 2–3, parallel | open |
| 015 | — | [Re-run benchmark on a quiet box](issues/015-rerun-benchmark-quiet.md) | open |
| 016 | — | [CatBoost GPU `eval_metric`](issues/016-catboost-gpu-eval-metric.md) | open |

**Phase 2 ordering note.** XGBoost GPU (007) comes before the other families: at 27 s/fold it is
4–8× cheaper than anything else here and is immune to the CPU contention, so it is the cheapest way
to price the new feature blocks. Budget LightGBM and CatBoost members deliberately — two or three
each, not ten.
