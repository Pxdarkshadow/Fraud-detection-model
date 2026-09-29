"""Fraud Detection Pipeline — LightGBM + XGBoost + SHAP explainability

Single file: load -> features -> split -> train -> compare -> evaluate -> benchmark

Dataset: data/fraud_merged.csv (Sparkov + BankSim + FDH Handbook, ~90k rows, 35% fraud).
  - Human-readable features: amount, category, merchant, location, time, demographics.
  - Threshold selection: fitted on VALIDATION only. Test set never touches threshold logic.

Explanations:
  - SHAP TreeExplainer for per-transaction feature contributions.
  - Benchmarked on a single transaction to keep runtime bounded.
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
    CATEGORICAL_COLS,
    MODEL_DIR,
    MODEL_PATH,
    THRESHOLD_PATH,
    XGB_MODEL_PATH,
    XGB_THRESHOLD_PATH,
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
    """Enrich df with log-amount and distance, convert categorical types for LGBM."""
    df = df.copy()

    if "amount" in df.columns:
        df["amount_log1p"] = np.log1p(df["amount"])

    if all(x in df.columns for x in ["lat", "long", "merch_lat", "merch_long"]):
        # ponytail: naive euclidean to avoid haversine overhead. Fine for local distances.
        dlat = df["lat"] - df["merch_lat"]
        dlon = (df["long"] - df["merch_long"]) * np.cos(np.radians(df["lat"].fillna(0)))
        df["distance_km"] = np.sqrt(dlat**2 + dlon**2) * 111.0
        # fill unknown locations with median dataset distance to avoid NaNs blowing up predict
        df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# 3. CHRONOLOGICAL SPLIT (on tx_datetime)
# ---------------------------------------------------------------------------

def chrono_split(df):
    """Chronologically split merged dataset.

    The merged CSV contains a ``tx_datetime`` column (ISO‑8601 strings).
    We sort by this timestamp, then allocate the newest portion to the test set,
    the next slice to a validation band (which is further stratified), and the
    remainder to the training set.
    """
    # Ensure datetime type for reliable ordering.
    df = df.copy()
    df["tx_datetime"] = pd.to_datetime(df["tx_datetime"], errors="coerce")
    df = df.sort_values("tx_datetime").reset_index(drop=True)

    max_idx = len(df)
    test_cutoff = int(max_idx * (1 - TEST_RATIO))
    val_cutoff = int(max_idx * (1 - TEST_RATIO - VAL_RATIO))

    train = df.iloc[:val_cutoff].copy()
    band = df.iloc[val_cutoff:test_cutoff].copy()
    test = df.iloc[test_cutoff:].copy()

    # Stratified holdout *within* the chronological validation band.
    _, val = train_test_split(
        band,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=band[TARGET] if TARGET in band.columns else None,
    )
    val = val.copy()
    train = pd.concat([train, band.drop(val.index)])

    print(f"  Train: {len(train):,} ({train[TARGET].mean():.4%} fraud)")
    print(f"  Val:   {len(val):,} ({val[TARGET].mean():.4%} fraud)")
    print(f"  Test:  {len(test):,} ({test[TARGET].mean():.4%} fraud)")
    return train, val, test


# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------

def train_lightgbm(train, val, feature_cols):
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        # dataset is 35% fraud — no artificial pos-weight needed
        "seed": RANDOM_SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(300),
            lgb.log_evaluation(100),
        ],
    )
    print(f"  LightGBM best iteration: {model.best_iteration}")
    return model


def train_xgboost(train, val, feature_cols):
    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]

    X_train = _xgb_encode(X_train)
    X_val = _xgb_encode(X_val)

    # dataset is 35% fraud — no artificial pos-weight needed
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=300,
        verbose_eval=100,
    )
    best_iter = model.best_iteration if model.best_iteration is not None else "?"
    print(f"  XGBoost best iteration: {best_iter}")
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
# 6. EVALUATE (applies a PRE-FITTED threshold)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    auc: float
    auc_pr: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion: np.ndarray


def _binary_prf1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # confusion_matrix ordering: [[tn, fp],[fn,tp]] for binary labels 0/1
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, cm


def _xgb_encode(X):
    """Integer-encode category columns for XGBoost DMatrix."""
    X = X.copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category").cat.codes
    return X


def predict_proba(model, X):
    # LightGBM Booster works directly with DataFrames.
    # XGBoost Booster needs integer-encoded DMatrix.
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(_xgb_encode(X)))
    return model.predict(X)


def evaluate(model, X, y_true, threshold, label="Test"):
    y_prob = predict_proba(model, X)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    precision, recall, f1, cm = _binary_prf1(y_true, y_pred)

    print(f"\n{'=' * 50}")
    print(f"{label}")
    print(f"{'=' * 50}")
    print(f"AUC:            {auc:.4f}")
    print(f"AUC-PR:         {auc_pr:.4f}")
    print(f"Threshold:      {threshold:.4f} (fitted on validation)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    try:
        print(
            "\nClassification report:\n"
            + classification_report(
                y_true,
                y_pred,
                target_names=["Legit", "Fraud"],
                digits=4,
            )
        )
    except Exception:
        pass

    return EvalResult(
        auc=float(auc),
        auc_pr=float(auc_pr),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=float(threshold),
        confusion=cm,
    )


def print_side_by_side(val_results, test_results):
    for split_name, split_results in [("Validation", val_results), ("Test", test_results)]:
        print(f"\n{'#' * 80}")
        print(f"{split_name}: LightGBM vs XGBoost")
        print(f"{'#' * 80}")

        rows = [
            ("ROC-AUC", "auc"),
            ("AUC-PR", "auc_pr"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
        ]

        lm, xgbm = split_results["LightGBM"], split_results["XGBoost"]
        for title, field in rows:
            print(
                f"{title:<12}  LightGBM: {getattr(lm, field):.4f} | XGBoost: {getattr(xgbm, field):.4f}"
            )

        print(f"Threshold    LightGBM: {lm.threshold:.4f} | XGBoost: {xgbm.threshold:.4f}")

        print("\nConfusion Matrix (actual 0/1):")
        print("LightGBM:\n" + str(lm.confusion))
        print("XGBoost:\n" + str(xgbm.confusion))
# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (LightGBM gain)
# ---------------------------------------------------------------------------

def print_feature_importance_lightgbm(model, feature_cols, top_n=15):
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:top_n]
    print(f"\nTop {top_n} features (LightGBM gain):")
    for i in order:
        print(f"  {feature_cols[i]:<25} {imp[i]:>12.1f}")


# ---------------------------------------------------------------------------
# 8. LATENCY + SHAP EXPLAINABILITY
# ---------------------------------------------------------------------------

def benchmark_latency(model, X_test, n_runs=200):
    row = X_test.iloc[[0]].copy()
    # warm-up
    for _ in range(10):
        predict_proba(model, row)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_proba(model, row)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"\nLatency ({n_runs} runs, single row):")
    print(f"  p50:  {np.percentile(times, 50):.2f} ms")
    print(f"  p95:  {np.percentile(times, 95):.2f} ms")
    print(f"  p99:  {np.percentile(times, 99):.2f} ms")
    print(f"  mean: {times.mean():.2f} ms")


def shap_top_k(model, X, feature_cols, top_k=5):
    import shap

    # TreeExplainer initialization can be non-trivial; keep per-call bounded.
    explainer = shap.TreeExplainer(model)

    shap_vals = explainer.shap_values(X, check_additivity=False)
    # Binary classifiers often return an array, but handle list outputs too.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    shap_vals = np.asarray(shap_vals)

    abs_vals = np.abs(shap_vals)
    k = min(top_k, abs_vals.shape[1])

    # Top-k by absolute contribution.
    idx = np.argpartition(-abs_vals, kth=k - 1, axis=1)[:, :k]
    # Sort within each row by abs value.
    row_abs = np.take_along_axis(abs_vals, idx, axis=1)
    order = np.argsort(row_abs, axis=1)[:, ::-1]
    idx_sorted = np.take_along_axis(idx, order, axis=1)

    explanations = []
    for i in range(shap_vals.shape[0]):
        top_feats = [feature_cols[j] for j in idx_sorted[i]]
        top_vals = [float(shap_vals[i, j]) for j in idx_sorted[i]]
        explanations.append({"features": top_feats, "shap_values": top_vals})

    return explanations


def benchmark_shap(model, X_test, feature_cols, top_k=5):
    row = X_test.iloc[[0]].copy()
    # inference time (separate from explanation time)
    t0 = time.perf_counter()
    _ = predict_proba(model, row)
    inference_ms = (time.perf_counter() - t0) * 1000

    # explanation time
    t0 = time.perf_counter()
    _ = shap_top_k(model, row[feature_cols], feature_cols, top_k=top_k)
    explanation_ms = (time.perf_counter() - t0) * 1000

    print(f"\nSHAP latency (single row, top-{top_k}):")
    print(f"  inference_ms:   {inference_ms:.2f} ms")
    print(f"  explanation_ms: {explanation_ms:.2f} ms")


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

        # Train both models on the same splits.
        lgb_model = train_lightgbm(train, val, FEATURES)
        xgb_model = train_xgboost(train, val, FEATURES)

        # Fit thresholds on validation ONLY (per-model).
        lgb_val_prob = predict_proba(lgb_model, val[FEATURES])
        xgb_val_prob = predict_proba(xgb_model, val[FEATURES])
        lgb_threshold = fit_threshold(val[TARGET], lgb_val_prob)
        xgb_threshold = fit_threshold(val[TARGET], xgb_val_prob)
        print(f"\n  Validation threshold (LightGBM max-F1): {lgb_threshold:.4f}")
        print(f"  Validation threshold (XGBoost  max-F1): {xgb_threshold:.4f}")

        # Evaluate val + test with their frozen thresholds.
        val_res_lgb = evaluate(
            lgb_model, val[FEATURES], val[TARGET], lgb_threshold, label="Validation (LightGBM)"
        )
        val_res_xgb = evaluate(
            xgb_model, val[FEATURES], val[TARGET], xgb_threshold, label="Validation (XGBoost)"
        )

        test_res_lgb = evaluate(
            lgb_model, test[FEATURES], test[TARGET], lgb_threshold, label="Test (LightGBM)"
        )
        test_res_xgb = evaluate(
            xgb_model, test[FEATURES], test[TARGET], xgb_threshold, label="Test (XGBoost)"
        )

        print_side_by_side(
            val_results={"LightGBM": val_res_lgb, "XGBoost": val_res_xgb},
            test_results={"LightGBM": test_res_lgb, "XGBoost": test_res_xgb},
        )

        # Feature importance and latency (per-model).
        print_feature_importance_lightgbm(lgb_model, FEATURES)
        benchmark_latency(lgb_model, test[FEATURES])
        benchmark_latency(xgb_model, test[FEATURES])

        # SHAP speed check on a single transaction.
        # Note: explanation latency includes SHAP init + shap_values for a single row.
        benchmark_shap(lgb_model, test[FEATURES], FEATURES, top_k=5)

        # Save artifacts.
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        lgb_model.save_model(str(MODEL_PATH))
        THRESHOLD_PATH.write_text(f"{lgb_threshold:.6f}")
        print(f"\nLightGBM model saved to {MODEL_PATH}")
        print(f"LightGBM threshold saved to {THRESHOLD_PATH} ({lgb_threshold:.6f})")

        xgb_model.save_model(str(XGB_MODEL_PATH))
        XGB_THRESHOLD_PATH.write_text(f"{xgb_threshold:.6f}")
        print(f"XGBoost model saved to {XGB_MODEL_PATH}")
        print(f"XGBoost threshold saved to {XGB_THRESHOLD_PATH} ({xgb_threshold:.6f})")

        # Optional: show a tiny explanation example for the LightGBM model.
        try:
            ex = shap_top_k(
                lgb_model,
                test[FEATURES].iloc[[0]],
                FEATURES,
                top_k=5,
            )[0]
            print("\nExample SHAP (LightGBM, first test row):")
            for f, v in zip(ex["features"], ex["shap_values"]):
                print(f"  {f:<10} {v:+.6f}")
        except Exception as e:
            print(f"\nSHAP example skipped: {e}")


# ---------------------------------------------------------------------------
# 10. MAIN (backward compat — direct execution)
# ---------------------------------------------------------------------------

def main():
    FraudPipeline().run()


if __name__ == "__main__":
    main()
