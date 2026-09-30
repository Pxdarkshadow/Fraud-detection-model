"""Shared constants for the fraud detection pipeline.

All paths are relative to the project root.
Dataset: data/fraud_merged.csv (Sparkov + BankSim + FDH Handbook, merged & rebalanced).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fraud_merged.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "fraud_model.txt"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

# ---------------------------------------------------------------------------
# Features & target
# ---------------------------------------------------------------------------
TARGET = "is_fraud"
CLASS_COL = "is_fraud"

# Raw inputs before engineering
RAW_NUMERIC = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

# Categorical features — LightGBM native categoricals
CATEGORICAL_FEATURES = [
    "category",
    "merchant",
    "gender",
    "age_bucket",
    "city",
    "state",
    "channel",
    "source",
]

INPUT_COLS = RAW_NUMERIC + CATEGORICAL_FEATURES

# Engineered model features
NUMERIC_FEATURES = RAW_NUMERIC + [
    "amount_log1p",
    "distance_km",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TEST_RATIO = 0.15
VAL_RATIO = 0.30
RANDOM_SEED = 42

CATEGORICAL_COLS = CATEGORICAL_FEATURES
