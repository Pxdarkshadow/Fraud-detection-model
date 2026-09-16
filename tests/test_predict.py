"""Minimal tests for predict module — plain asserts, runnable directly.

Model-loading tests are guarded on data/creditcard.csv + models/fraud_model.txt:
they SKIP now and arm themselves once retrain lands.
"""

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd

# Add src/ to path so we can import the package
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "src"))

from fraud_detection.predict import INPUT_COLS, validate_input, predict, default_threshold
from fraud_detection.config import MODEL_PATH, DATA_PATH

# Also resolve paths relative to project root for file checks
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = str(MODEL_PATH)
DATA = str(DATA_PATH)


def expect_valueerror(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


# --- 1. Rejection (no model needed) ---
# (a) frame missing Amount -> ValueError
no_amount = pd.DataFrame({c: [0.0] for c in INPUT_COLS if c != "Amount"})
assert expect_valueerror(lambda: validate_input(no_amount)), "missing Amount must raise ValueError"
print("PASS: missing Amount rejected")

# (b) NaN in V1 -> ValueError
nan_v1 = pd.DataFrame({c: [0.0] for c in INPUT_COLS})
nan_v1.loc[0, "V1"] = float("nan")
assert expect_valueerror(lambda: validate_input(nan_v1)), "NaN in V1 must raise ValueError"
print("PASS: NaN in V1 rejected")

# --- 2. CLI bad-input rejects (validate runs before model load) ---
# Run predict module as: python -m fraud_detection.predict --in bad_input.json
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
bad_path = os.path.join(project_root, "bad_input.json")
with open(bad_path, "w") as f:
    json.dump([{"Time": 0, "V1": 1.0}], f)
cli = subprocess.run(
    [sys.executable, "-m", "fraud_detection.predict", "--in", bad_path],
    capture_output=True,
    text=True,
    cwd=os.path.join(project_root, "src"),
)
os.remove(bad_path)
assert cli.returncode != 0, f"CLI should exit non-zero on missing column, rc={cli.returncode}"
msg = (cli.stderr or cli.stdout).strip()
assert msg and msg.count("\n") == 0, f"CLI should print a one-line error, got: {msg!r}"
print(f"PASS: CLI exits non-zero with one-line error: {msg}")

# --- 3. Threshold default ---
if os.path.isfile(MODEL) and os.path.isfile(str(MODEL_PATH.parent / "threshold.txt")):
    from fraud_detection.config import THRESHOLD_PATH
    with open(THRESHOLD_PATH) as f:
        t = float(f.read().strip())
    assert 0.0 <= t <= 1.0, f"threshold {t} out of [0,1]"
    assert default_threshold() == t, "default threshold must match models/threshold.txt"
    print(f"PASS: default threshold {t} read from models/threshold.txt")
else:
    print("SKIP: models/threshold.txt absent; default threshold stays 0.5")

# --- 4. Model tests (armed after retrain) ---
if os.path.isfile(MODEL) and os.path.isfile(DATA):
    df = pd.read_csv(DATA, nrows=5000)
    labels = df["Class"].values
    assert (labels == 1).any(), "no fraud rows in the nrows=5000 sample"
    probs = predict(df)["fraud_probability"].values

    assert np.isfinite(probs).all(), "probabilities must be finite"
    assert ((probs >= 0) & (probs <= 1)).all(), "probabilities must be in [0,1]"
    assert len(set(probs.tolist())) > 1, "model must not return identical probabilities"

    mean1 = probs[labels == 1].mean()
    mean0 = probs[labels == 0].mean()
    margin = mean1 - mean0
    print(f"PASS: mean p(fraud|Class=1)={mean1:.4f} vs Class=0={mean0:.4f}, margin={margin:.4f}")
    assert margin > 0.10, f"honesty check failed: margin {margin:.4f} <= 0.10"
    print("PASS: model honesty check (margin > 0.10)")
else:
    print("SKIP: model tests await retrain (need data/creditcard.csv + models/fraud_model.txt)")

print("\nAll tests passed.")
