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
DATASET_DIR = PROJECT_ROOT / "dataset"
SYNTHETIC_DATA_PATH = DATASET_DIR / "synthetic_fraud_transactions.csv"
CREDITCARD_DATA_PATH = DATASET_DIR / "creditcard.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "fraud_model.txt"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"
ISO_FOREST_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"

# ---------------------------------------------------------------------------
# Features & target
# ---------------------------------------------------------------------------
TARGET = "is_fraud"
CLASS_COL = "is_fraud"

# Isolation Forest universal numeric features
ISO_FOREST_NUMERIC_COLS = [
    "amount",
    "amount_log1p",
    "hour_sin",
    "hour_cos",
]

# Hybrid features incorporating Stage 1 anomaly scores into LightGBM
GBM_HYBRID_FEATURES = ISO_FOREST_NUMERIC_COLS + [
    "iso_forest_score",
    "iso_forest_outlier_flag",
]

# Risk classification tiers
RISK_LEVELS = [
    (0.80, "Critical", "RED", "Block Transaction"),
    (0.65, "High", "ORANGE", "Manual Review"),
    (0.45, "Medium", "YELLOW", "Approve with Monitoring"),
    (0.00, "Low", "GREEN", "Approve"),
]

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
