"""Fraud Detection Pipeline — ULB Real Credit-Card Transactions

Single file: load -> features -> split -> train -> evaluate -> benchmark

Dataset: ULB/Kaggle credit-card fraud (284,807 rows, 492 frauds ~0.17%).
  - V1..V28 are PCA-transformed anonymized features.
  - Time = seconds elapsed since first transaction in the dataset (not wall-clock).
  - Real fraud signals are far subtler than PaySim's synthetic balance-draining pattern.

Threshold selection: fitted on VALIDATION only. Test set never touches threshold logic.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from .config import (
    DATA_PATH,
    TEST_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    TARGET,
    FEATURES,
    MODEL_DIR,
    MODEL_PATH,
    THRESHOLD_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows | Fraud rate: {df[TARGET].mean():.4%}")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df):
    """Return features only — no derived columns."""
    return df


# ---------------------------------------------------------------------------
# 3. CHRONOLOGICAL SPLIT (on Time = elapsed seconds)
# ---------------------------------------------------------------------------
def chrono_split(df):
    max_time = df["Time"].max()
    test_cutoff = max_time * (1 - TEST_RATIO)
    val_cutoff = max_time * (1 - TEST_RATIO - VAL_RATIO)

    train = df[df["Time"] < val_cutoff].copy()
    val = df[(df["Time"] >= val_cutoff) & (df["Time"] < test_cutoff)].copy()
    test = df[df["Time"] >= test_cutoff].copy()

    print(f"  Train: {len(train):,} ({train[TARGET].mean():.4%} fraud)")
    print(f"  Val:   {len(val):,} ({val[TARGET].mean():.4%} fraud)")
    print(f"  Test:  {len(test):,} ({test[TARGET].mean():.4%} fraud)")
    return train, val, test


# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
def train_model(train, val, feature_cols):
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": RANDOM_SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# 5. THRESHOLD FITTING (validation only)
# ---------------------------------------------------------------------------
def fit_threshold(y_true, y_prob):
    """Find threshold that maximizes F1. Called once on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


# ---------------------------------------------------------------------------
# 6. EVALUATE (applies a PRE-FITTED threshold — never selects one)
# ---------------------------------------------------------------------------
def evaluate(model, X, y_true, threshold, label="Test"):
    """Evaluate with a fixed threshold. Threshold was selected on val, not here."""
    y_prob = model.predict(X)
    auc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"AUC:            {auc:.4f}")
    print(f"AUC-PR:         {auc_pr:.4f}")
    print(f"Threshold:      {threshold:.4f} (fitted on validation)")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit', 'Fraud'])}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return auc, auc_pr


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
def print_feature_importance(model, feature_cols, top_n=15):
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:top_n]
    print(f"\nTop {top_n} features (gain):")
    for i in order:
        print(f"  {feature_cols[i]:<25} {imp[i]:>12.1f}")


# ---------------------------------------------------------------------------
# 8. LATENCY
# ---------------------------------------------------------------------------
def benchmark_latency(model, X_test, n_runs=1000):
    row = X_test.iloc[[0]].copy()
    for _ in range(50):
        model.predict(row)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"\nLatency ({n_runs} runs, single row):")
    print(f"  p50:  {np.percentile(times, 50):.2f} ms")
    print(f"  p95:  {np.percentile(times, 95):.2f} ms")
    print(f"  p99:  {np.percentile(times, 99):.2f} ms")
    print(f"  mean: {times.mean():.2f} ms")


# ---------------------------------------------------------------------------
# 9. PIPELINE CLASS
# ---------------------------------------------------------------------------
class FraudPipeline:
    """Encapsulates the full train-evaluate-save pipeline."""

    def run(self):
        df = load_data()
        df = engineer_features(df)
        train, val, test = chrono_split(df)
        del df

        model = train_model(train, val, FEATURES)

        # Fit threshold on validation ONLY
        val_prob = model.predict(val[FEATURES])
        threshold = fit_threshold(val[TARGET], val_prob)
        print(f"  Validation threshold (max F1): {threshold:.4f}")

        # Evaluate val + test with that fixed threshold
        evaluate(model, val[FEATURES], val[TARGET], threshold, "Validation")
        evaluate(model, test[FEATURES], test[TARGET], threshold, "Test")
        print_feature_importance(model, FEATURES)
        benchmark_latency(model, test[FEATURES])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))
        THRESHOLD_PATH.write_text(f"{threshold:.6f}")
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Threshold saved to {THRESHOLD_PATH} ({threshold:.6f})")


# ---------------------------------------------------------------------------
# 10. MAIN (backward compat — direct execution)
# ---------------------------------------------------------------------------
def main():
    FraudPipeline().run()


if __name__ == "__main__":
    main()
