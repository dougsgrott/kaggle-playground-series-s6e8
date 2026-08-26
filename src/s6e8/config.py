"""Competition-wide constants. Single source of truth for schema and the fold contract."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("S6E8_DATA_DIR", PROJECT_ROOT / "data"))
OOF_DIR = Path(os.environ.get("S6E8_OOF_DIR", PROJECT_ROOT / "oof"))
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
FOLDS_NPY = DATA_DIR / "folds.npy"

ID_COL = "id"
TARGET = "addicted_label"

N_TRAIN = 691_369
N_TEST = 296_302

# --- the frozen fold contract -------------------------------------------------
# StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train[TARGET])
# over train.csv in ORIGINAL FILE ROW ORDER. The whole public OOF ecosystem uses this,
# so any borrowed member drops straight in. Never redefine these inline.
N_SPLITS = 5
FOLD_SEED = 42

NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
FEATURES = NUM_COLS + CAT_COLS

CAT_LEVELS = {
    "gender": ["Female", "Male", "Other"],
    "stress_level": ["High", "Low", "Medium"],
    "academic_work_impact": ["No", "Yes"],
}

# Columns with a fractional part; the integer-valued ones (age, notifications_per_day,
# app_opens_per_day) have no decimals to read.
FRACTIONAL_COLS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "weekend_screen_time",
]

# daily_screen_time_hours >= social_media_hours + gaming_hours + work_study_hours,
# exact on 100.000% of train+test rows. See docs/data-notes.md.
BUDGET_TOTAL = "daily_screen_time_hours"
BUDGET_PARTS = ["social_media_hours", "gaming_hours", "work_study_hours"]

# --- the noise floor, measured on this box 2026-08-26 --------------------------
# 55 XGBoost-GPU fits: 6 partition seeds x 6 model seeds on the starter feature block.
# Raw numbers and every cell: docs/noise_floor.json. Method: scripts/noise_floor.py.
#
# Two distinct quantities, and the smaller one is NOT the one to reach for by habit:
#
#   SIGMA_PARTITION  how far the pooled CV number moves when the PARTITION changes.
#                    Gates comparisons against a CV computed on some other split.
#   SIGMA_DELTA      how far an A-vs-B delta moves under the null on the FROZEN
#                    partition with both arms re-seeded. Gates every row in
#                    docs/experiments.md, which all hold the partition at seed 42.
#
# Model seed beats partition seed here 2:1 -- pooling 691,369 rows makes the split
# barely matter, while subsample/colsample redraw in all five models. So repeating CV
# across partitions is the WRONG way to buy precision on an ablation; averaging model
# seeds is the right way, and cuts SIGMA_DELTA by sqrt(n).
SIGMA_PARTITION = 1.9e-5
SIGMA_MODEL = 3.9e-5
SIGMA_DELTA = 5.5e-5

# The default gate. An ablation on the frozen folds must clear 2 * SIGMA_DELTA
# = 1.1e-4 to have measured anything; two identical runs were observed 1.0e-4 apart.
NOISE_FLOOR = SIGMA_DELTA

# Mean per-fold AUC range inside a single run. 22x SIGMA_DELTA, and NOT uncertainty:
# on the frozen partition it is dominated by fold 3 being genuinely easier (+0.00097)
# and fold 0 harder (-0.00061), a fixed property of the split that reproduced across
# all six model seeds. Never quote it as an error bar.
FOLD_RANGE_TYPICAL = 1.2e-3
