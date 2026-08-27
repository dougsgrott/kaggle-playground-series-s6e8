"""Member definitions, registered by name for scripts/train_member.py.

Each entry is a callable taking (train, test, y) and returning a `FitPredict` closure
for `cv.run_member`. Modelling libraries are imported lazily inside the builders so
`s6e8.runtime.configure_threads()` can run before any OpenMP pool is sized.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from s6e8 import config as C
from s6e8.runtime import assert_xgboost_gpu, n_threads

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def wrap(fn):
        REGISTRY[name] = fn
        return fn
    return wrap


# --------------------------------------------------------------------------------
# xgb_baseline -- a faithful port of the community starter.
#
# Source: analysis/nb_clean/cdeotte__simple-xgb-starter.py
#
# Ported AS WRITTEN, feature block and hyperparameters both. This member exists to
# prove the path (folds -> member -> OOF contract -> submission) and to place the
# first point on the CV->LB line. It is deliberately NOT the measured-positive
# feature set -- most of these ratios measured negative in the corpus and issue 002
# replaces them. Do not tune it; its value is that it is comparable to a published
# number (~0.9640 OOF / ~0.9655 LB).
# --------------------------------------------------------------------------------

STARTER_XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.035,
    max_depth=7,
    min_child_weight=15,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=3.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    early_stopping_rounds=150,
    random_state=C.FOLD_SEED,
    verbosity=0,
)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def starter_features(df: pd.DataFrame) -> pd.DataFrame:
    """The starter's engineered block, verbatim from cdeotte__simple-xgb-starter.py."""
    out = df.copy()

    base_predictors = [c for c in out.columns if c != C.ID_COL]
    out["missing_count"] = out[base_predictors].isna().sum(axis=1).astype("float32")

    component_cols = ["social_media_hours", "gaming_hours", "work_study_hours"]
    out["screen_component_total"] = out[component_cols].sum(axis=1, min_count=len(component_cols))
    out["time_accounting_residual"] = (
        out["daily_screen_time_hours"]
        - out["social_media_hours"]
        - out["gaming_hours"]
        - out["work_study_hours"]
    )
    out["weekend_lift"] = out["weekend_screen_time"] - out["daily_screen_time_hours"]
    out["weekend_ratio"] = _safe_divide(out["weekend_screen_time"], out["daily_screen_time_hours"])

    out["social_share_of_daily"] = _safe_divide(out["social_media_hours"], out["daily_screen_time_hours"])
    out["gaming_share_of_daily"] = _safe_divide(out["gaming_hours"], out["daily_screen_time_hours"])
    out["work_study_share_of_daily"] = _safe_divide(out["work_study_hours"], out["daily_screen_time_hours"])

    out["sleep_minus_screen"] = out["sleep_hours"] - out["daily_screen_time_hours"]
    out["screen_to_sleep_ratio"] = _safe_divide(out["daily_screen_time_hours"], out["sleep_hours"])

    out["opens_per_screen_hour"] = _safe_divide(out["app_opens_per_day"], out["daily_screen_time_hours"])
    out["notifications_per_screen_hour"] = _safe_divide(out["notifications_per_day"], out["daily_screen_time_hours"])
    out["notifications_per_open"] = _safe_divide(out["notifications_per_day"], out["app_opens_per_day"])
    out["engagement_events"] = out["notifications_per_day"] + out["app_opens_per_day"]

    out = out.replace([np.inf, -np.inf], np.nan)

    for c in C.CAT_COLS:
        out[c] = out[c].fillna("(missing)").astype(str)
    return out


def starter_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot the three categoricals over train+test so columns line up exactly."""
    tr = starter_features(train.drop(columns=[C.TARGET])).drop(columns=[C.ID_COL])
    te = starter_features(test).drop(columns=[C.ID_COL])

    combined = pd.concat([tr, te], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=C.CAT_COLS, dummy_na=False, dtype=np.float32)
    combined = combined.astype(np.float32)

    X = combined.iloc[: len(tr)].reset_index(drop=True)
    X_test = combined.iloc[len(tr):].reset_index(drop=True)
    assert X.columns.equals(X_test.columns)
    return X, X_test


@register("xgb_baseline")
def build_xgb_baseline(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray):
    import xgboost as xgb

    print(f"  {assert_xgboost_gpu()}", flush=True)
    X, X_test = starter_matrices(train, test)
    print(f"  design matrix {X.shape} / {X_test.shape}  "
          f"{X.memory_usage(deep=True).sum() / 1024 ** 2:.0f} MB", flush=True)

    params = dict(STARTER_XGB_PARAMS, device="cuda", n_jobs=n_threads())
    best_iters: list[int] = []

    def fit_predict(fold: int, train_idx: np.ndarray, valid_idx: np.ndarray):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[train_idx], y[train_idx],
            eval_set=[(X.iloc[valid_idx], y[valid_idx])],
            verbose=False,
        )
        best_iters.append(int(model.best_iteration) + 1)
        return (model.predict_proba(X.iloc[valid_idx])[:, 1],
                model.predict_proba(X_test)[:, 1])

    fit_predict.extra = {"best_iters": best_iters}
    return fit_predict


# --------------------------------------------------------------------------------
# xgb_features -- the measured-positive feature set from issue 002.
#
# Blocks that PAID, in the order they were priced (docs/experiments.md):
#   lattice   +0.001609   the decimal lattice -- the largest single block by 2.6x
#   te        +0.000740   nested stringified target encoding
#   budget    +0.000626   the accounting-identity residual and composition
#   freq      +0.000609   transductive value counts over 987,671 rows
#   impute    +0.000193   XGB-imputed numerics alongside the raw NaN columns
#   decimals  +0.000112   printed decimal places, mode-filled (see features.py)
#
# `cat_isna` is EXCLUDED: it measured -0.000007 against a null sd of 1.8e-05 over three
# seeds per arm -- UNRESOLVED, not positive. An unproven block is not worth three
# columns in every member downstream.
#
# Unlike xgb_baseline this is a real member, not a calibration point, so early stopping
# is back and the round cap is raised: issue 004 found the starter's 3000 barely bound
# (best iterations 2521-2818).
# --------------------------------------------------------------------------------

FEATURE_BLOCKS = ("raw", "budget", "lattice", "decimals", "freq", "impute")

FEATURES_XGB_PARAMS = dict(
    STARTER_XGB_PARAMS, n_estimators=6000, enable_categorical=True,
)


def _xgb_view(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray,
              blocks=FEATURE_BLOCKS, use_te: bool = True, **overrides):
    """Build a FitPredict over one feature view. The Phase-2 XGBoost workhorse.

    Views differ by which BLOCKS they carry, not by seed: the admission rule kills
    members whose rank correlation to an existing one exceeds ~0.99, and reseeding the
    same feature set has been measured at +0.00002 blend contribution. See docs/issues/007.
    """
    import xgboost as xgb

    from s6e8.features import NestedTargetEncoder, build_static

    print(f"  {assert_xgboost_gpu()}", flush=True)
    fs = build_static(train, test, n_jobs=n_threads())
    cols = fs.columns(blocks)
    X, X_test = fs.train[cols], fs.test[cols]
    encoder = NestedTargetEncoder(train, test, y) if use_te else None
    n_te = len(encoder.names) if encoder else 0
    print(f"  blocks {list(blocks)}  te={use_te}", flush=True)
    print(f"  design matrix {X.shape[0]:,} x {X.shape[1]} static + {n_te} target-encoded",
          flush=True)

    params = dict(FEATURES_XGB_PARAMS, device="cuda", n_jobs=n_threads(), **overrides)
    best_iters: list[int] = []

    def fit_predict(fold: int, train_idx: np.ndarray, valid_idx: np.ndarray):
        Xa = X.iloc[train_idx].reset_index(drop=True)
        Xb = X.iloc[valid_idx].reset_index(drop=True)
        Xt = X_test.reset_index(drop=True)
        if encoder is not None:
            # Rebuilt per fold: the encoder reads y, so it can never be precomputed.
            e_tr, e_va, e_te = encoder.build(train_idx, valid_idx)
            Xa = pd.concat([Xa, e_tr], axis=1)
            Xb = pd.concat([Xb, e_va], axis=1)
            Xt = pd.concat([Xt, e_te], axis=1)

        model = xgb.XGBClassifier(**params)
        model.fit(Xa, y[train_idx], eval_set=[(Xb, y[valid_idx])], verbose=False)
        best_iters.append(int(model.best_iteration) + 1)
        return (model.predict_proba(Xb)[:, 1], model.predict_proba(Xt)[:, 1])

    fit_predict.extra = {"best_iters": best_iters, "blocks": list(blocks), "te": use_te,
                         **{k: v for k, v in overrides.items()}}
    return fit_predict


@register("xgb_features")
def build_xgb_features(train, test, y):
    """The flagship: every block that paid in issue 002, plus nested target encoding."""
    return _xgb_view(train, test, y)


@register("xgb_no_te")
def build_xgb_no_te(train, test, y):
    """Drops the target-reading mechanism entirely.

    Value separation then has to come from `lattice` and `freq` alone. Those are
    substitutes for target encoding (issue 002), so this should stay strong while
    making different mistakes -- the shape a stack member wants.
    """
    return _xgb_view(train, test, y, use_te=False)


@register("xgb_te_only")
def build_xgb_te_only(train, test, y):
    """The opposite bet: target encoding WITHOUT the lattice.

    `lattice` was the largest single block (+0.001609) and `te` the second (+0.000740),
    and they do the same job by different means. Removing the lattice forces the model
    onto the target-statistics route and should decorrelate it from the flagship more
    than any hyperparameter change could.
    """
    return _xgb_view(train, test, y, blocks=("raw", "budget", "freq", "impute"))


@register("xgb_deep")
def build_xgb_deep(train, test, y):
    """Same features, different bias/variance point -- EXPECTED TO FAIL ADMISSION.

    Run once on purpose. "Same algorithm, different params -> weight 0.000" and
    "tree depth 9-13 costs up to -0.0011" are both corpus claims this repo has not
    tested itself, and a member that earns weight 0.000 is a cheap, recorded negative.
    """
    return _xgb_view(train, test, y, max_depth=10, min_child_weight=40,
                     learning_rate=0.03)
