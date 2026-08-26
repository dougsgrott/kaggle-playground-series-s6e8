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

# Measured on this data: std of repeated-5-fold means over 3 partition seeds.
# A delta below this has measured nothing.
NOISE_FLOOR = 5e-5
