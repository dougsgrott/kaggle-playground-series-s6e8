"""Feature blocks for S6E8 — only the ones with a measured reason to exist.

Two kinds of block, and the distinction is the whole architecture:

**Static blocks are target-free**, so they are computed ONCE over train+test together and
sliced per fold. Transductive use of the test features is not leakage — no label is
involved — and it is what makes the frequency counts and the imputers stronger than
their train-only versions.

**The target encoder is fold-dependent** and is rebuilt inside every outer fold with its
own inner 5-fold, because it reads `y`. It never appears in the static frame.

Block inventory, in descending measured value (corpus figures; local numbers land in
docs/experiments.md):

| block | what | corpus Δ |
|---|---|---|
| `te` | nested stringified target encoding, all 12 columns | +0.0023 |
| `impute` | XGB-imputed numerics ALONGSIDE the raw NaN columns | +0.0012 |
| `budget` | the accounting-identity residual and composition | +0.0005 |
| `freq` | transductive value counts over 987,671 rows | +0.0003 |
| `lattice` | decimal lattice: fractional part + first digit | +0.0001 |
| `decimals` | printed decimal-place count, read from raw CSV text | part of the above |
| `cat_isna` | the three CATEGORICAL missing flags only | +0.0001 |

Sources: `analysis/nb_clean/tomasa2__s6e8-what-moved-the-score-and-what-didn-t.py`
(the ablation of record and the nested `build_enc` scheme),
`analysis/nb_clean/dariushafshar__s6e8-what-actually-helps.py` (decimal places from raw
text), `analysis/nb_clean/najiama__single-lgbm-model-lb-0-96990-cv-0-96862.py`
(transductive counts).

Deliberately NOT built, each measured negative — see docs/issues/002 for the full list:
numeric `is_missing` flags (−0.00001, and they identify the train/test split), the generic
behavioural ratios, `wk_ratio`, `week_total`, `sleep_deficit`, pairwise and
multi-resolution TE, TE smoothing above 10, the second decimal digit, DAE features,
pseudo-labelling, and the original 7,500-row dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from s6e8 import config as C
from s6e8.data import read_decimal_places, string_levels

# Every block except `te`, which is fold-dependent and lives in NestedTargetEncoder.
STATIC_BLOCKS = ("raw", "budget", "cat_isna", "lattice", "decimals", "freq",
                 "impute")

# Columns the accounting identity relates. Exact on 100.000% of train+test rows:
# daily >= social + gaming + work_study. Axis-aligned trees cannot construct the
# residual themselves, which is why it is worth handing them.
_ACC_PARTS = C.BUDGET_PARTS
_ACC_TOTAL = C.BUDGET_TOTAL

TE_SMOOTH = 10.0
TE_INNER_SPLITS = 5
TE_INNER_SEED = 0

# One XGB regressor per numeric column, fit on train+test. Target-free, so this is
# transductive preprocessing rather than leakage. Params from tomasa2.
_IMPUTER_PARAMS = dict(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=20, tree_method="hist",
                       enable_categorical=True, verbosity=0)


@dataclass
class FeatureSet:
    """Aligned train/test frames plus the block each column came from."""
    train: pd.DataFrame
    test: pd.DataFrame
    groups: dict[str, list[str]] = field(default_factory=dict)

    def columns(self, blocks) -> list[str]:
        missing = [b for b in blocks if b not in self.groups]
        if missing:
            raise KeyError(f"unknown block(s) {missing}; have {sorted(self.groups)}")
        return [c for b in blocks for c in self.groups[b]]

    def select(self, blocks) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = self.columns(blocks)
        return self.train[cols].copy(), self.test[cols].copy()

    def describe(self) -> str:
        rows = [f"  {b:10s} {len(self.groups[b]):3d}  {', '.join(self.groups[b][:4])}"
                + ("..." if len(self.groups[b]) > 4 else "")
                for b in self.groups]
        return (f"FeatureSet  train {self.train.shape}  test {self.test.shape}\n"
                + "\n".join(rows))


# ================================================================================
# Static blocks
# ================================================================================

def _raw_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """The 12 columns as given. Categoricals stay native — never one-hot the full frame
    (7.8 GB box, see CLAUDE.md); LightGBM/XGBoost/CatBoost all take `category` dtype."""
    out = {}
    for c in C.NUM_COLS:
        out[c] = (train[c].astype(np.float32), test[c].astype(np.float32))
    levels = {c: pd.CategoricalDtype(categories=C.CAT_LEVELS[c]) for c in C.CAT_COLS}
    for c in C.CAT_COLS:
        out[c] = (train[c].astype(levels[c]), test[c].astype(levels[c]))
    return out


def _budget_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """Composition derived from the accounting identity, and nothing else.

    `resid` (a.k.a. other_screen) is the latent unallocated screen time; it has standalone
    AUC 0.765 on complete rows and no tree can build it from axis-aligned splits. The
    fractions express the same budget in shares. The generic behavioural ratios
    (notifications-per-open, weekend ratio, weekly total) are excluded on measurement,
    not taste — see docs/issues/002.
    """
    out = {}

    def pair(name, fn):
        out[name] = (fn(train).astype(np.float32), fn(test).astype(np.float32))

    # min_count forces NaN unless every part is present. A plain sum() would skip NaNs
    # and silently define the residual as "daily minus whatever happened to be there",
    # which is a different and much weaker variable. This exact bug passed a check once
    # already -- see docs/issues/001.
    def parts(df):
        return df[_ACC_PARTS].sum(axis=1, min_count=len(_ACC_PARTS))

    pair("bd_resid", lambda d: d[_ACC_TOTAL] - parts(d))
    pair("bd_parts", parts)
    pair("bd_leisure", lambda d: d[_ACC_TOTAL] - d["work_study_hours"])
    pair("bd_resid_frac", lambda d: (d[_ACC_TOTAL] - parts(d)) / d[_ACC_TOTAL])
    pair("bd_social_frac", lambda d: d["social_media_hours"] / d[_ACC_TOTAL])
    pair("bd_gaming_frac", lambda d: d["gaming_hours"] / d[_ACC_TOTAL])
    pair("bd_work_frac", lambda d: d["work_study_hours"] / d[_ACC_TOTAL])
    pair("bd_awake_frac", lambda d: d[_ACC_TOTAL] / (24.0 - d["sleep_hours"]))
    pair("bd_free_time",
         lambda d: 24.0 - d["sleep_hours"] - d[_ACC_TOTAL] - d["work_study_hours"])
    return out


def _cat_isna_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """Missing flags for the THREE CATEGORICAL columns only.

    The numeric ones are excluded deliberately: they measured −0.00001 on CV and, because
    missing rates differ between train and test in every column, they hand the model a
    train/test discriminator. Adversarial validation is 0.5649 with NaNs and 0.4985 on
    complete rows -- the missingness *is* the shift. See docs/data-notes.md.
    """
    return {f"na_{c}": (train[c].isna().astype(np.int8), test[c].isna().astype(np.int8))
            for c in C.CAT_COLS}


def _lattice_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """The decimal lattice: fractional part and first decimal digit.

    Values are quantised to 1 or 2 decimals (5 rows in the whole corpus excepted), so
    `rint(v * 100)` recovers the integer lattice exactly -- floor(v * 10) would not,
    since 1.8 stores as 1.7999...

    NaN in, NaN out: a missing value has no lattice position, and inventing one would
    smuggle the missingness pattern in through the back door. That is the whole reason
    this is separate from `decimals` -- see that block's warning.
    """
    out = {}
    for c in C.FRACTIONAL_COLS:
        ctr = np.rint(train[c].to_numpy(np.float64) * 100.0)
        cte = np.rint(test[c].to_numpy(np.float64) * 100.0)
        out[f"frac_{c}"] = (pd.Series((ctr % 100) / 100.0, dtype=np.float32),
                            pd.Series((cte % 100) / 100.0, dtype=np.float32))
        out[f"d1_{c}"] = (pd.Series((ctr // 10) % 10, dtype=np.float32),
                          pd.Series((cte // 10) % 10, dtype=np.float32))
    return out


def _decimals_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """Printed decimal-place count per field, read from the RAW CSV TEXT.

    float64 cannot tell you that "1.80" was written with two decimals and "1.8" with
    one, so this has to be read before pandas parses the field.

    **Missing is filled with the mode (2), not left as NaN**, and that is the whole
    reason this block is separate from `lattice`. `read_decimal_places` returns NaN
    wherever the field is missing, and that NaN pattern is bit-identical to the raw
    missingness pattern -- so the natural version of this block is a numeric missingness
    flag with a decimal bit stapled on, and numeric missingness flags are on the
    do-not-build list precisely because they raise CV and lower LB.

    Measured on this data (docs/issues/002):

      * the decimal bit does NOT shift between train and test -- the 2dp rate differs by
        |z| < 1.7 in all six columns, so the bit itself is not a train/test discriminator;
      * the bit does carry real signal -- P(y | 1dp) = 0.6821 vs P(y | 2dp) = 0.7121 on
        `daily_screen_time_hours`, a +0.0300 gap;
      * the NaN channel is pure missingness and duplicates what the raw column already
        exposes.

    Filling with the mode keeps the first two and discards the third: missing rows merge
    into the majority group instead of forming their own splittable one.
    """
    nd_tr = read_decimal_places(C.TRAIN_CSV, C.FRACTIONAL_COLS)
    nd_te = read_decimal_places(C.TEST_CSV, C.FRACTIONAL_COLS)
    assert len(nd_tr) == len(train) and len(nd_te) == len(test)
    out = {}
    for col in nd_tr.columns:
        mode = float(nd_tr[col].mode().iloc[0])
        out[col] = (nd_tr[col].fillna(mode).astype(np.float32),
                    nd_te[col].fillna(mode).astype(np.float32))
        assert out[col][0].notna().all() and out[col][1].notna().all()
    return out


def _freq_block(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.Series]:
    """Transductive value counts over all 987,671 train+test rows.

    Target-free, so pooling test is legitimate and strictly better: a value seen 3 times
    in train and 2 in test is a 5-count, and the tail lookups stop being noise.

    Keys go through `string_levels`, NOT `astype(str)`. On pandas 3.x the new str dtype
    preserves NA, so `astype(str)` drops every missing row out of `value_counts` and the
    missing rows silently map to 0 instead of forming their own (large) level. The source
    notebook uses the naive form and would be wrong on this environment.
    """
    ltr = string_levels(train, C.FEATURES)
    lte = string_levels(test, C.FEATURES)
    out = {}
    for c in C.FEATURES:
        counts = pd.concat([ltr[c], lte[c]], ignore_index=True).value_counts()
        assert counts.sum() == len(train) + len(test), (
            f"{c}: counts cover {counts.sum():,} of {len(train) + len(test):,} rows")
        out[f"fq_{c}"] = (ltr[c].map(counts).astype(np.float32),
                          lte[c].map(counts).astype(np.float32))
    return out


def _impute_block(train: pd.DataFrame, test: pd.DataFrame, seed: int = C.FOLD_SEED,
                  device: str = "cuda", n_jobs: int = 2,
                  verbose: bool = True) -> dict[str, pd.Series]:
    """One XGB regressor per numeric column, fit on train+test with no target involved.

    Imputed columns are added ALONGSIDE the raw NaN-bearing ones, never replacing them:
    replacing throws away the missingness pattern, which the corpus measured as worth
    +0.0012 to keep. Ported from tomasa2's `impute`.
    """
    import xgboost as xgb

    n = len(train)
    full = pd.concat([train[C.FEATURES], test[C.FEATURES]], ignore_index=True)
    X = full.copy()
    for c in C.CAT_COLS:
        X[c] = X[c].astype(pd.CategoricalDtype(categories=C.CAT_LEVELS[c]))

    out = {}
    for col in C.NUM_COLS:
        obs = X[col].notna().to_numpy()
        feats = [c for c in C.FEATURES if c != col]
        filled = full[col].to_numpy(np.float64).copy()
        if (~obs).sum():
            model = xgb.XGBRegressor(**_IMPUTER_PARAMS, device=device, n_jobs=n_jobs,
                                     random_state=seed)
            model.fit(X.loc[obs, feats], X.loc[obs, col])
            filled[~obs] = model.predict(X.loc[~obs, feats])
        if verbose:
            print(f"    imputed {col:26s} {(~obs).sum():>7,} of {len(X):,} rows",
                  flush=True)
        s = pd.Series(filled, dtype=np.float32)
        out[f"imp_{col}"] = (s.iloc[:n].reset_index(drop=True),
                            s.iloc[n:].reset_index(drop=True))
    return out


_BUILDERS = {
    "raw": _raw_block,
    "budget": _budget_block,
    "cat_isna": _cat_isna_block,
    "lattice": _lattice_block,
    "decimals": _decimals_block,
    "freq": _freq_block,
    "impute": _impute_block,
}


CACHE_DIR = C.PROJECT_ROOT / "data" / "features"

# Bump whenever a block's DEFINITION changes. The cache is keyed on the block list, and
# the block list alone cannot notice that `decimals` switched from NaN-carrying to
# mode-filled -- a stale hit would silently score the old features under the new name.
FEATURE_VERSION = "v2"


def _cache_paths(blocks) -> tuple:
    tag = f"{FEATURE_VERSION}_" + "-".join(blocks)
    return (CACHE_DIR / f"static_{tag}_train.parquet",
            CACHE_DIR / f"static_{tag}_test.parquet",
            CACHE_DIR / f"static_{tag}_groups.json")


def build_static(train: pd.DataFrame, test: pd.DataFrame, blocks=STATIC_BLOCKS,
                 verbose: bool = True, cache: bool = True, **kwargs) -> FeatureSet:
    """Build the target-free blocks over train+test and return an aligned FeatureSet.

    Cached to data/features/ because `impute` costs minutes and every Phase 2 member
    would otherwise pay it again. The cache key is the block list; it is NOT keyed on
    the imputer seed or device, so pass cache=False when varying those.
    """
    import json

    blocks = tuple(blocks)
    p_tr, p_te, p_g = _cache_paths(blocks)
    if cache and p_tr.exists() and p_te.exists() and p_g.exists():
        fs = FeatureSet(train=pd.read_parquet(p_tr), test=pd.read_parquet(p_te),
                        groups=json.loads(p_g.read_text()))
        assert len(fs.train) == len(train) and len(fs.test) == len(test)
        if verbose:
            print(f"  static cache hit  {p_tr.name}", flush=True)
            print(fs.describe(), flush=True)
        return fs

    fs = _build_static_uncached(train, test, blocks, verbose, **kwargs)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fs.train.to_parquet(p_tr, index=False)
        fs.test.to_parquet(p_te, index=False)
        p_g.write_text(json.dumps(fs.groups, indent=2))
        if verbose:
            print(f"  cached -> {p_tr.parent}", flush=True)
    return fs


def _build_static_uncached(train: pd.DataFrame, test: pd.DataFrame, blocks,
                           verbose: bool = True, **kwargs) -> FeatureSet:
    tr_cols: dict[str, pd.Series] = {}
    te_cols: dict[str, pd.Series] = {}
    groups: dict[str, list[str]] = {}

    for block in blocks:
        if block not in _BUILDERS:
            raise KeyError(f"unknown block {block!r}; have {sorted(_BUILDERS)}")
        if verbose:
            print(f"  block {block}", flush=True)
        kw = kwargs if block == "impute" else {}
        made = _BUILDERS[block](train, test, **kw)
        groups[block] = list(made)
        for name, (a, b) in made.items():
            tr_cols[name] = np.asarray(a) if isinstance(a, pd.Series) else a
            te_cols[name] = np.asarray(b) if isinstance(b, pd.Series) else b
            # Categorical dtype does not survive np.asarray; keep those as Series.
            if isinstance(a, pd.Series) and str(a.dtype) == "category":
                tr_cols[name], te_cols[name] = a.reset_index(drop=True), b.reset_index(drop=True)

    X = pd.DataFrame(tr_cols)
    X_test = pd.DataFrame(te_cols)
    assert len(X) == len(train) and len(X_test) == len(test)
    assert list(X.columns) == list(X_test.columns)
    fs = FeatureSet(train=X, test=X_test, groups=groups)
    if verbose:
        print(fs.describe(), flush=True)
    return fs


# ================================================================================
# The fold-dependent block: nested stringified target encoding
# ================================================================================

class NestedTargetEncoder:
    """Stringified target encoding, rebuilt inside every outer fold.

    The scheme, from tomasa2's `build_enc`:

      * outer TRAIN rows get encodings from an inner 5-fold — each inner-out block is
        encoded by maps fitted on the other four, so no row ever sees its own label;
      * outer VALID and TEST rows get encodings from maps fitted on the whole outer
        training portion, which is the most data available to them honestly.

    Keys are exact printed values, not bins: with 167–1,437 distinct values per column
    the value itself is a usable lookup key, and this is the same mechanism a large
    `max_bin` exploits — which is why the two are substitutes and their published gains
    must never be added (docs/issues/002).

    Missing is an EXPLICIT level via `string_levels`, so the ~14–19% of rows with a NaN
    in a given column get their own statistic instead of the base rate.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
                 cols=None, smooth: float = TE_SMOOTH,
                 inner_splits: int = TE_INNER_SPLITS, inner_seed: int = TE_INNER_SEED):
        self.cols = list(cols if cols is not None else C.FEATURES)
        self.y = np.asarray(y)
        self.smooth = smooth
        self.inner_splits = inner_splits
        self.inner_seed = inner_seed
        self.levels_train = string_levels(train, self.cols)
        self.levels_test = string_levels(test, self.cols)
        self.names = [f"te_{c}" for c in self.cols]

    def _maps(self, levels: pd.DataFrame, y: np.ndarray) -> tuple[dict, float]:
        prior = float(y.mean())
        maps = {}
        for c in self.cols:
            g = (pd.DataFrame({"lv": levels[c].to_numpy(), "y": y})
                 .groupby("lv")["y"].agg(["count", "mean"]))
            # string_levels gives missing its own level, so every row must be covered.
            # Without that this assertion is what catches the pandas 3.x NA trap.
            assert int(g["count"].sum()) == len(y), (
                f"{c}: groupby covers {int(g['count'].sum()):,} of {len(y):,} rows -- "
                "a level key leaked NA; check string_levels")
            maps[c] = ((g["count"] * g["mean"] + self.smooth * prior)
                       / (g["count"] + self.smooth)).astype(np.float32)
        return maps, prior

    def _apply(self, levels: pd.DataFrame, maps: dict, prior: float) -> pd.DataFrame:
        return pd.DataFrame(
            {f"te_{c}": levels[c].map(maps[c]).astype(np.float32).fillna(prior).to_numpy()
             for c in self.cols})

    def build(self, train_idx: np.ndarray, valid_idx: np.ndarray
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (encodings for train_idx, for valid_idx, for test) for one outer fold."""
        from sklearn.model_selection import StratifiedKFold

        y_tr = self.y[train_idx]
        L = self.levels_train.iloc[train_idx].reset_index(drop=True)

        holder = np.zeros((len(train_idx), len(self.names)), dtype=np.float32)
        inner = StratifiedKFold(n_splits=self.inner_splits, shuffle=True,
                                random_state=self.inner_seed)
        seen = np.zeros(len(train_idx), dtype=bool)
        for i_in, i_out in inner.split(np.zeros(len(train_idx)), y_tr):
            maps, prior = self._maps(L.iloc[i_in], y_tr[i_in])
            holder[i_out] = self._apply(
                L.iloc[i_out].reset_index(drop=True), maps, prior).to_numpy()
            seen[i_out] = True
        assert seen.all(), "inner folds did not cover every outer-training row"

        maps, prior = self._maps(L, y_tr)
        return (pd.DataFrame(holder, columns=self.names),
                self._apply(self.levels_train.iloc[valid_idx].reset_index(drop=True),
                            maps, prior),
                self._apply(self.levels_test, maps, prior))
