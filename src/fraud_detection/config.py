"""Shared constants for the fraud detection pipeline.

All paths are relative to the project root (D:\fraud detection model).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.txt"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

# ---------------------------------------------------------------------------
# Features & target
# ---------------------------------------------------------------------------
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
INPUT_COLS = FEATURES  # alias used by predict module
CLASS_COL = "Class"

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TEST_RATIO = 0.15
VAL_RATIO = 0.15
RANDOM_SEED = 42
CATEGORICAL_COLS = []
