"""Fraud Detection Pipeline — LightGBM + SHAP explainability

Single file: load -> features -> split -> train -> evaluate -> benchmark

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
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

from .config import (
    DATA_PATH,
    SYNTHETIC_DATA_PATH,
    CREDITCARD_DATA_PATH,
    TEST_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    TARGET,
    FEATURES,
    CATEGORICAL_COLS,
    ISO_FOREST_NUMERIC_COLS,
    GBM_HYBRID_FEATURES,
    MODEL_DIR,
    MODEL_PATH,
    THRESHOLD_PATH,
    ISO_FOREST_MODEL_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def load_data(nrows: int | None = 100_000) -> pd.DataFrame:
    """Load transaction data.
    
    1. Checks for senior intern's merged dataset: data/fraud_merged.csv
    2. Falls back to dataset/synthetic_fraud_transactions.csv
    3. Falls back to dataset/creditcard.csv
    """
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df):,} rows from {DATA_PATH.name} | Fraud rate: {df[TARGET].mean():.4%}")
        return df

    if SYNTHETIC_DATA_PATH.exists():
        print(f"Loading synthetic dataset from {SYNTHETIC_DATA_PATH.name}...")
        raw = pd.read_csv(SYNTHETIC_DATA_PATH, nrows=nrows)
        df = pd.DataFrame()
        df["amount"] = pd.to_numeric(raw["amount"], errors="coerce").astype("float32")

        if "hour_of_day" in raw.columns:
            df["hour"] = pd.to_numeric(raw["hour_of_day"], errors="coerce").fillna(12.0).astype("float32")
        else:
            df["hour"] = 12.0

        if "day_of_week" in raw.columns:
            df["day_of_week"] = pd.to_numeric(raw["day_of_week"], errors="coerce").fillna(2).astype("int8")
        else:
            df["day_of_week"] = 2

        if "is_weekend" in raw.columns:
            df["is_weekend"] = pd.to_numeric(raw["is_weekend"], errors="coerce").fillna(0).astype("int8")
        else:
            df["is_weekend"] = 0

        for col in ["lat", "long", "merch_lat", "merch_long"]:
            if col in raw.columns:
                df[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")
            else:
                df[col] = np.nan

        df["category"] = raw["merchant_category_desc"].astype(str).str.strip().str.lower() if "merchant_category_desc" in raw.columns else "unknown"
        df["merchant"] = raw["merchant_name"].astype(str).str.strip().str.lower() if "merchant_name" in raw.columns else "unknown"
        df["gender"] = raw["customer_gender"].astype(str).str.strip().str.upper() if "customer_gender" in raw.columns else "U"
        df["age_bucket"] = raw["customer_income_bracket"].astype(str).str.strip() if "customer_income_bracket" in raw.columns else "U"
        df["city"] = raw["customer_city"].astype(str).str.strip() if "customer_city" in raw.columns else "unknown"
        df["state"] = raw["customer_state"].astype(str).str.strip() if "customer_state" in raw.columns else "unknown"
        df["channel"] = raw["channel"].astype(str).str.strip().str.lower() if "channel" in raw.columns else "pos"
        df["source"] = "synthetic"

        df["tx_datetime"] = pd.to_datetime(raw["timestamp"], errors="coerce") if "timestamp" in raw.columns else pd.to_datetime("2026-01-01")
        target_col = "is_fraud" if "is_fraud" in raw.columns else ("isFraud" if "isFraud" in raw.columns else None)
        df[TARGET] = pd.to_numeric(raw[target_col], errors="coerce").fillna(0).astype("int8") if target_col else 0

        print(f"Loaded {len(df):,} synthetic rows | Fraud rate: {df[TARGET].mean():.4%}")
        return df

    if CREDITCARD_DATA_PATH.exists():
        print(f"Loading credit card dataset from {CREDITCARD_DATA_PATH.name}...")
        raw = pd.read_csv(CREDITCARD_DATA_PATH, nrows=nrows)
        df = pd.DataFrame()
        df["amount"] = pd.to_numeric(raw["Amount"], errors="coerce").astype("float32")
        time_sec = pd.to_numeric(raw["Time"], errors="coerce").fillna(0.0)
        df["hour"] = ((time_sec % 86400.0) / 3600.0).astype("float32")
        df["day_of_week"] = 2
        df["is_weekend"] = 0
        df["tx_datetime"] = pd.to_datetime("2026-01-01 00:00:00") + pd.to_timedelta(time_sec, unit="s")

        for col in ["lat", "long", "merch_lat", "merch_long"]:
            df[col] = np.nan
        for col in ["category", "merchant", "gender", "age_bucket", "city", "state", "channel"]:
            df[col] = "unknown"
        df["source"] = "creditcard"

        target_col = "Class" if "Class" in raw.columns else ("is_fraud" if "is_fraud" in raw.columns else None)
        df[TARGET] = pd.to_numeric(raw[target_col], errors="coerce").fillna(0).astype("int8") if target_col else 0

        print(f"Loaded {len(df):,} credit card rows | Fraud rate: {df[TARGET].mean():.4%}")
        return df

    raise FileNotFoundError(f"No dataset found at {DATA_PATH} or {SYNTHETIC_DATA_PATH}")


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df):
    """Enrich df with log-amount, cyclical hour sin/cos, distance, and categories."""
    df = df.copy()

    if "amount" in df.columns:
        df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0.0)).astype("float32")

    if "hour" in df.columns:
        hour = pd.to_numeric(df["hour"], errors="coerce").fillna(12.0)
    elif "hour_of_day" in df.columns:
        hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(12.0)
    else:
        hour = pd.Series(12.0, index=df.index, dtype="float32")

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")

    if all(x in df.columns for x in ["lat", "long", "merch_lat", "merch_long"]):
        # naive euclidean to avoid haversine overhead. Fine for local distances.
        dlat = df["lat"] - df["merch_lat"]
        dlon = (df["long"] - df["merch_long"]) * np.cos(np.radians(df["lat"].fillna(0)))
        df["distance_km"] = np.sqrt(dlat**2 + dlon**2) * 111.0
        df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median() if df["distance_km"].notna().any() else 0.0)
    elif "distance_km" not in df.columns:
        df["distance_km"] = 0.0

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
    df = df.copy()
    if "tx_datetime" in df.columns:
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
        stratify=band[TARGET] if TARGET in band.columns and band[TARGET].nunique() > 1 else None,
    )
    val = val.copy()
    train = pd.concat([train, band.drop(val.index)])

    print("\n--- Chronological Split ---")
    print(f"  Train: {len(train):,} ({train[TARGET].mean():.4%} fraud)")
    print(f"  Val:   {len(val):,} ({val[TARGET].mean():.4%} fraud)")
    print(f"  Test:  {len(test):,} ({test[TARGET].mean():.4%} fraud)")
    return train, val, test


# ---------------------------------------------------------------------------
# 4. STAGE 1: ISOLATION FOREST ANOMALY DETECTION
# ---------------------------------------------------------------------------

def train_isolation_forest(train_df: pd.DataFrame) -> tuple[IsolationForest, list[str]]:
    """Fit Isolation Forest on universal features (amount, log-amount, hour_sin, hour_cos)."""
    iso_cols = [c for c in ISO_FOREST_NUMERIC_COLS if c in train_df.columns]
    X_train = train_df[iso_cols].fillna(0.0).astype("float32")

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model, iso_cols


def add_isolation_scores(df: pd.DataFrame, model: IsolationForest, iso_cols: list[str]) -> pd.DataFrame:
    """Enrich dataframe with `iso_forest_score` and `iso_forest_outlier_flag`."""
    df = df.copy()
    X = df[iso_cols].fillna(0.0).astype("float32")
    df["iso_forest_score"] = (-model.score_samples(X)).astype("float32")
    df["iso_forest_outlier_flag"] = (model.predict(X) == -1).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 5. STAGE 2: LIGHTGBM TRAINING WITH IN-FOLD SMOTE
# ---------------------------------------------------------------------------

def train_lightgbm(train, val, feature_cols):
    available_features = [c for c in feature_cols if c in train.columns and c in val.columns]
    X_train = train[available_features].copy()
    y_train = train[TARGET].copy()
    X_val = val[available_features].copy()
    y_val = val[TARGET].copy()

    orig_fraud = int((y_train == 1).sum())
    orig_normal = int((y_train == 0).sum())
    print(f"\n[SMOTE] Original training distribution: Normal={orig_normal:,} | Fraud={orig_fraud:,}")

    # Check if SMOTE can be applied (on purely numeric data with severe class imbalance < 30%)
    numeric_features = [c for c in available_features if pd.api.types.is_numeric_dtype(X_train[c])]
    has_categoricals = any(c in CATEGORICAL_COLS for c in available_features)

    if not has_categoricals and orig_fraud > 5 and (orig_fraud / (orig_normal + 1e-5)) < 0.30 and len(numeric_features) == len(available_features):
        fill_vals = X_train.median(numeric_only=True).fillna(0.0)
        X_train_clean = X_train.fillna(fill_vals)
        X_val = X_val.fillna(fill_vals)

        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_clean, y_train)
        new_fraud = int((y_train_resampled == 1).sum())
        new_normal = int((y_train_resampled == 0).sum())
        print(f"[SMOTE] Augmented training distribution: Normal={new_normal:,} | Fraud={new_fraud:,}")
        dtrain = lgb.Dataset(X_train_resampled, y_train_resampled)
    else:
        cat_cols = [c for c in CATEGORICAL_COLS if c in available_features]
        dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)

    cat_cols_val = [c for c in CATEGORICAL_COLS if c in available_features]
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 48,
        "max_depth": -1,
        "min_child_samples": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": RANDOM_SEED,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(150),
            lgb.log_evaluation(50),
        ],
    )
    print(f"  LightGBM best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# 6. THRESHOLD FITTING & EVALUATION
# ---------------------------------------------------------------------------

def fit_threshold(y_true, y_prob):
    """Find threshold that maximizes F1. Called once on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


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
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, cm


def predict_proba(model, X):
    return model.predict(X)


def evaluate(model, X, y_true, threshold, label="Test"):
    y_prob = predict_proba(model, X)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    precision, recall, f1, cm = _binary_prf1(np.asarray(y_true), y_pred)

    print(f"\n{'=' * 50}")
    print(f"{label} Evaluation")
    print(f"{'=' * 50}")
    print(f"AUC:            {auc:.4f}")
    print(f"AUC-PR:         {auc_pr:.4f}")
    print(f"Threshold:      {threshold:.4f} (fitted on validation)")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
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


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE & LATENCY BENCHMARKING
# ---------------------------------------------------------------------------

def print_feature_importance_lightgbm(model, feature_cols, top_n=15):
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:top_n]
    print(f"\nTop {min(top_n, len(order))} features (LightGBM gain):")
    for i in order:
        if i < len(feature_cols):
            print(f"  {feature_cols[i]:<25} {imp[i]:>12.1f}")


def benchmark_latency(model, X_test, n_runs=200):
    row = X_test.iloc[[0]].copy()
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

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    shap_vals = np.asarray(shap_vals)
    abs_vals = np.abs(shap_vals)
    k = min(top_k, abs_vals.shape[1])

    idx = np.argpartition(-abs_vals, kth=k - 1, axis=1)[:, :k]
    row_abs = np.take_along_axis(abs_vals, idx, axis=1)
    order = np.argsort(row_abs, axis=1)[:, ::-1]
    idx_sorted = np.take_along_axis(idx, order, axis=1)

    explanations = []
    for i in range(shap_vals.shape[0]):
        top_feats = [feature_cols[j] for j in idx_sorted[i]]
        top_vals = [float(shap_vals[i, j]) for j in idx_sorted[i]]
        explanations.append({"features": top_feats, "shap_values": top_vals})

    return explanations


# ---------------------------------------------------------------------------
# 8. PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

class FraudPipeline:
    """Encapsulates the full train-evaluate-save hybrid pipeline."""

    def run(self):
        df = load_data()
        df = engineer_features(df)
        train, val, test = chrono_split(df)
        del df

        # Stage 1: Isolation Forest
        print("\n" + "=" * 50)
        print("STAGE 1: Unsupervised Isolation Forest")
        print("=" * 50)
        iso_model, iso_cols = train_isolation_forest(train)
        train = add_isolation_scores(train, iso_model, iso_cols)
        val = add_isolation_scores(val, iso_model, iso_cols)
        test = add_isolation_scores(test, iso_model, iso_cols)
        print(f"Isolation Forest fitted on features: {iso_cols}")

        # Stage 2: LightGBM
        feature_cols = [c for c in (FEATURES + ["iso_forest_score", "iso_forest_outlier_flag"]) if c in train.columns and c in val.columns]
        print("\n" + "=" * 50)
        print(f"STAGE 2: Supervised LightGBM ({len(feature_cols)} features)")
        print("=" * 50)
        model = train_lightgbm(train, val, feature_cols)

        # Fit threshold on validation ONLY
        val_prob = predict_proba(model, val[feature_cols])
        threshold = fit_threshold(val[TARGET], val_prob)
        print(f"\nValidation threshold (max-F1): {threshold:.4f}")

        evaluate(model, val[feature_cols], val[TARGET], threshold, label="Validation")
        evaluate(model, test[feature_cols], test[TARGET], threshold, label="Test")

        print_feature_importance_lightgbm(model, feature_cols)
        benchmark_latency(model, test[feature_cols])

        # Save artifacts
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))
        THRESHOLD_PATH.write_text(f"{threshold:.6f}")
        joblib.dump(iso_model, ISO_FOREST_MODEL_PATH)

        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Isolation Forest saved to {ISO_FOREST_MODEL_PATH}")
        print(f"Threshold saved to {THRESHOLD_PATH} ({threshold:.6f})")

        try:
            ex = shap_top_k(model, test[feature_cols].iloc[[0]], feature_cols, top_k=5)[0]
            print("\nExample SHAP (first test row):")
            for f, v in zip(ex["features"], ex["shap_values"]):
                print(f"  {f:<20} {v:+.6f}")
        except Exception as e:
            print(f"\nSHAP example skipped: {e}")


# ---------------------------------------------------------------------------
# 9. MAIN (backward compat — direct execution)
# ---------------------------------------------------------------------------

def main():
    FraudPipeline().run()


if __name__ == "__main__":
    main()
