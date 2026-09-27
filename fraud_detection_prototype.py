from __future__ import annotations
#!/usr/bin/env python3
import os
import warnings
from pathlib import Path
import argparse
import math
import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASET_DIR = BASE_DIR / "dataset"


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _resolve_data_path(env_name: str, default_name: str, fallback_names: tuple[str, ...]) -> Path:
    candidates = []
    env_value = os.environ.get(env_name)
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(DATASET_DIR / default_name)
    candidates.extend(DATASET_DIR / name for name in fallback_names)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a usable dataset for {env_name}. Tried: {tried}")


TRAIN_DATA_PATH = _resolve_data_path(
    "FRAUD_TRAIN_PATH",
    "synthetic_fraud_transactions.csv",
    ("Fraud.csv", "creditcard.csv"),
)
VALID_DATA_PATH = _resolve_data_path(
    "FRAUD_VALID_PATH",
    "Fraud.csv",
    ("creditcard.csv", "synthetic_fraud_transactions.csv"),
)

TRAIN_MAX_ROWS = _env_int("FRAUD_TRAIN_MAX_ROWS", 100_000)
VALID_MAX_ROWS = _env_int("FRAUD_VALID_MAX_ROWS", 100_000)

TARGET_COL = "isFraud"
SOURCE_COL = "source"

TYPE_MAP = {
    "cash_in": "cash_in",
    "cash-in": "cash_in",
    "cash_out": "cash_out",
    "cash-out": "cash_out",
    "debit": "debit",
    "payment": "payment",
    "transfer": "transfer",
    "w": "purchase",
    "c": "cash_out",
    "r": "refund",
    "h": "purchase",
    "s": "service",
    "chip transaction": "purchase",
    "swipe transaction": "purchase",
    "online transaction": "purchase",
}

SYNTHETIC_NUMERIC_COLS = [
    "customer_age",
    "customer_credit_score",
    "account_age_days",
    "merchant_risk_score",
    "merchant_years_in_business",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "txn_count_last_7d",
    "amount_sum_last_24h",
    "distinct_merchants_last_24h",
    "distinct_countries_last_24h",
    "avg_txn_amount_customer_90d",
    "std_txn_amount_customer_90d",
    "time_since_last_txn_minutes",
    "distance_from_home_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

SYNTHETIC_CATEGORICAL_COLS = [
    "customer_gender",
    "customer_income_bracket",
    "account_type",
    "customer_country",
    "customer_state",
    "customer_city",
    "merchant_category_code",
    "merchant_category_desc",
    "merchant_country",
    "merchant_city",
    "channel",
    "entry_mode",
    "device_type",
    "device_os",
    "browser",
    "ip_country",
    "currency",
]

CREDITCARD_PCA_COLS = [f"principal_component_{i:02d}" for i in range(1, 29)]
CREDITCARD_RENAME_MAP = {
    "Time": "transaction_time_seconds",
    "Amount": "transaction_amount_usd",
    "Class": TARGET_COL,
    **{f"V{i}": CREDITCARD_PCA_COLS[i - 1] for i in range(1, 29)},
}

CAT_FEATURES = [
    "type",
    SOURCE_COL,
    "account_type",
    "customer_gender",
    "customer_income_bracket",
    "merchant_category_desc",
    "channel",
    "entry_mode",
    "device_type",
    "device_os",
    "browser",
    "currency",
    "ip_country",
]

DROP_COLS = [TARGET_COL]

SHARED_COLS = [
    SOURCE_COL,
    "amount",
    "type",
    "hour",
    "balance_before_orig",
    "balance_after_orig",
    "balance_before_dest",
    "balance_after_dest",
    "has_identity_info",
    TARGET_COL,
] + SYNTHETIC_NUMERIC_COLS + SYNTHETIC_CATEGORICAL_COLS + CREDITCARD_PCA_COLS

FRAUD_MODEL_NUMERIC_COLS = [
    "amount",
    "hour",
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "balance_delta_orig",
    "balance_delta_dest",
    "amount_to_balance_ratio",
    "missing_balance_info",
] + SYNTHETIC_NUMERIC_COLS

FRAUD_MODEL_CATEGORICAL_COLS = ["type"] + SYNTHETIC_CATEGORICAL_COLS

FRAUD_MODEL_FEATURE_COLS = FRAUD_MODEL_NUMERIC_COLS + FRAUD_MODEL_CATEGORICAL_COLS


def _empty_shared(n):
    frame = pd.DataFrame(index=range(n))
    for col in SHARED_COLS:
        frame[col] = np.nan
    return frame


def _coerce_numeric(series):
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0})
    return pd.to_numeric(mapped.fillna(series), errors="coerce").astype("float32")


def _coerce_category(series):
    return series.astype("string").str.strip().str.lower().fillna("missing").replace("<na>", "missing")


def load_synthetic(path=TRAIN_DATA_PATH, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    out = _empty_shared(len(df))
    out[SOURCE_COL] = "synthetic"
    out["amount"] = _coerce_numeric(df["amount"])
    out["hour"] = _coerce_numeric(df["hour_of_day"])
    transaction_type = _coerce_category(df["transaction_type"])
    out["type"] = transaction_type.map(TYPE_MAP).fillna(transaction_type)
    for col in SYNTHETIC_NUMERIC_COLS:
        if col in df.columns:
            out[col] = _coerce_numeric(df[col])
    for col in SYNTHETIC_CATEGORICAL_COLS:
        if col in df.columns:
            out[col] = _coerce_category(df[col])
    out[TARGET_COL] = _coerce_numeric(df["is_fraud"]).fillna(0).astype("int8")
    return out


def load_bank_fraud(path=VALID_DATA_PATH, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    out = _empty_shared(len(df))
    out[SOURCE_COL] = "fraud"
    out["amount"] = _coerce_numeric(df["amount"])
    out["hour"] = _coerce_numeric(df["step"]) % 24.0
    transaction_type = _coerce_category(df["type"])
    out["type"] = transaction_type.map(TYPE_MAP).fillna(transaction_type)
    out["balance_before_orig"] = _coerce_numeric(df["oldbalanceOrg"])
    out["balance_after_orig"] = _coerce_numeric(df["newbalanceOrig"])
    out["balance_before_dest"] = _coerce_numeric(df["oldbalanceDest"])
    out["balance_after_dest"] = _coerce_numeric(df["newbalanceDest"])
    out["has_identity_info"] = (~df[["nameOrig", "nameDest"]].isna().any(axis=1)).astype("int8")
    out[TARGET_COL] = _coerce_numeric(df["isFraud"]).fillna(0).astype("int8")
    return out


def load_dataset(path, nrows=None):
    columns = set(pd.read_csv(path, nrows=0).columns)
    if {"amount", "hour_of_day", "transaction_type", "is_fraud"}.issubset(columns):
        return load_synthetic(path=path, nrows=nrows)
    if {"amount", "step", "type", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"}.issubset(columns):
        return load_bank_fraud(path=path, nrows=nrows)
    if {"Time", "Amount", "Class"}.issubset(columns):
        return load_creditcard(path=path, nrows=nrows)
    raise ValueError(f"Unsupported dataset schema for {path}: {sorted(columns)}")


def load_creditcard(path=VALID_DATA_PATH, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    df = df.rename(columns=CREDITCARD_RENAME_MAP)
    out = _empty_shared(len(df))
    out[SOURCE_COL] = "creditcard"
    out["amount"] = _coerce_numeric(df["transaction_amount_usd"])
    out["hour"] = (_coerce_numeric(df["transaction_time_seconds"]) % 86400.0) / 3600.0
    out["type"] = "purchase"
    for col in CREDITCARD_PCA_COLS:
        if col in df.columns:
            out[col] = _coerce_numeric(df[col])
    out[TARGET_COL] = _coerce_numeric(df[TARGET_COL]).fillna(0).astype("int8")
    return out


def engineer_features(df):
    df = df.copy()

    df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))

    for col in ["balance_before_orig", "balance_after_orig", "balance_before_dest", "balance_after_dest"]:
        if col not in df.columns:
            df[col] = np.nan

    balances_present = df[["balance_before_orig", "balance_after_orig", "balance_before_dest", "balance_after_dest"]].notna().any().any()
    if balances_present:
        df["balance_delta_orig"] = df["balance_after_orig"] - df["balance_before_orig"]
        df["balance_delta_dest"] = df["balance_after_dest"] - df["balance_before_dest"]
        df["amount_to_balance_ratio"] = df["amount"] / (df["balance_before_orig"].abs() + 1.0)
        df["missing_balance_info"] = df["balance_before_orig"].isna().astype("int8")
    else:
        df["balance_delta_orig"] = np.nan
        df["balance_delta_dest"] = np.nan
        df["amount_to_balance_ratio"] = np.nan
        df["missing_balance_info"] = 0

    hour = df["hour"].fillna(df["hour"].median())
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df["type"] = df["type"].astype("object").fillna("other").astype("category")
    df[SOURCE_COL] = df[SOURCE_COL].astype("object").astype("category")
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("object").fillna("missing").astype("category")
    return df


def build_fraud_model_row(tx: dict) -> pd.DataFrame:
    timestamp = pd.Timestamp(tx["timestamp"])
    hour = timestamp.hour + timestamp.minute / 60.0
    day_of_week = timestamp.dayofweek
    weekend = int(day_of_week >= 5)
    rng = seeded_rng(tx["transaction_id"])

    row = {
        "amount": float(tx["amount"]),
        "hour": hour,
        "type": TYPE_MAP.get(tx["payment_method"].strip().lower(), "payment"),
        "customer_age": rng.randint(21, 78),
        "customer_credit_score": rng.randint(300, 850),
        "account_age_days": rng.randint(30, 3650),
        "merchant_risk_score": round(rng.uniform(0.0, 1.0), 4),
        "merchant_years_in_business": rng.randint(1, 25),
        "txn_count_last_1h": rng.randint(0, 8),
        "txn_count_last_24h": rng.randint(1, 25),
        "txn_count_last_7d": rng.randint(5, 120),
        "amount_sum_last_24h": round(rng.uniform(25.0, 12000.0), 2),
        "distinct_merchants_last_24h": rng.randint(1, 8),
        "distinct_countries_last_24h": rng.randint(1, 4),
        "avg_txn_amount_customer_90d": round(rng.uniform(20.0, 900.0), 2),
        "std_txn_amount_customer_90d": round(rng.uniform(5.0, 250.0), 2),
        "time_since_last_txn_minutes": round(rng.uniform(0.0, 2000.0), 2),
        "distance_from_home_km": round(rng.uniform(0.0, 2500.0), 2),
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "is_weekend": weekend,
        "customer_gender": rng.choice(["m", "f"]),
        "customer_income_bracket": rng.choice(["low", "mid", "high"]),
        "account_type": rng.choice(["checking", "savings", "credit"]),
        "customer_country": tx["location"].split(",")[-1].strip().lower(),
        "customer_state": rng.choice(["na", "ca", "tx", "ny", "il", "missing"]),
        "customer_city": tx["location"].split(",")[0].strip().lower(),
        "merchant_category_code": rng.choice(["5411", "5311", "5812", "5732", "7995"]),
        "merchant_category_desc": tx["merchant_category"].lower(),
        "merchant_country": tx["location"].split(",")[-1].strip().lower(),
        "merchant_city": tx["location"].split(",")[0].strip().lower(),
        "channel": rng.choice(["online", "mobile", "in_store"]),
        "entry_mode": rng.choice(["swipe", "chip", "contactless"]),
        "device_type": rng.choice(["mobile", "desktop", "tablet"]),
        "device_os": rng.choice(["ios", "android", "windows", "macos", "linux"]),
        "browser": rng.choice(["chrome", "safari", "edge", "firefox"]),
        "ip_country": tx["location"].split(",")[-1].strip().lower(),
        "currency": "usd",
    }

    row["balance_before_orig"] = np.nan
    row["balance_after_orig"] = np.nan
    row["balance_before_dest"] = np.nan
    row["balance_after_dest"] = np.nan
    row["has_identity_info"] = int(rng.random() < 0.7)
    row[SOURCE_COL] = "synthetic"

    frame = engineer_features(pd.DataFrame([row]))
    return frame[FRAUD_MODEL_FEATURE_COLS].copy()


def encode_fraud_features(frame: pd.DataFrame, fitted_columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(
        frame[FRAUD_MODEL_FEATURE_COLS].copy(),
        columns=FRAUD_MODEL_CATEGORICAL_COLS,
        dummy_na=False,
    )
    if fitted_columns is not None:
        encoded = encoded.reindex(columns=fitted_columns, fill_value=0)
    return encoded.astype("float32")


def train_fraud_classifier(train_df):
    features = train_df[FRAUD_MODEL_FEATURE_COLS].copy()
    target = train_df[TARGET_COL].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=target if target.nunique() > 1 else None,
    )

    X_train_encoded = encode_fraud_features(X_train)
    X_valid_encoded = encode_fraud_features(X_valid, fitted_columns=list(X_train_encoded.columns))

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train_encoded, y_train)

    valid_prob = model.predict_proba(X_valid_encoded)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_prob)
    valid_ap = average_precision_score(y_valid, valid_prob)

    feature_names = list(X_train_encoded.columns)
    importance_sample = permutation_importance(
        model,
        X_valid_encoded,
        y_valid,
        n_repeats=1,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
    )
    importances = importance_sample.importances_mean
    return model, feature_names, importances, list(X_train_encoded.columns)


train_df = engineer_features(load_dataset(TRAIN_DATA_PATH, nrows=TRAIN_MAX_ROWS))
valid_df = engineer_features(load_dataset(VALID_DATA_PATH, nrows=VALID_MAX_ROWS))

print("Loaded train source:", TRAIN_DATA_PATH.name, "rows:", len(train_df))
print("Loaded validation source:", VALID_DATA_PATH.name, "rows:", len(valid_df))
print("Feature-engineered train shape:", train_df.shape)
print("Feature-engineered validation shape:", valid_df.shape)


ISO_FOREST_PARAMS = dict(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

SYNTHETIC_ISO_FOREST_COLS = [
    "amount",
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "customer_age",
    "customer_credit_score",
    "account_age_days",
    "merchant_risk_score",
    "merchant_years_in_business",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "txn_count_last_7d",
    "amount_sum_last_24h",
    "distinct_merchants_last_24h",
    "distinct_countries_last_24h",
    "avg_txn_amount_customer_90d",
    "std_txn_amount_customer_90d",
    "time_since_last_txn_minutes",
    "distance_from_home_km",
    "balance_delta_orig",
    "balance_delta_dest",
    "amount_to_balance_ratio",
    "missing_balance_info",
]

CREDITCARD_ISO_FOREST_COLS = [
    "amount",
    "amount_log1p",
    "hour_sin",
    "hour_cos",
] + CREDITCARD_PCA_COLS


def get_iso_forest_matrix(df, feature_cols, fill_values=None, scale_values=None):
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values).fillna(0.0)
    if scale_values is None:
        q75 = X.quantile(0.75, numeric_only=True)
        q25 = X.quantile(0.25, numeric_only=True)
        scale_values = (q75 - q25).replace(0, 1.0).fillna(1.0)
    X = (X - fill_values).divide(scale_values.replace(0, 1.0))
    return X.astype("float32"), cols, fill_values, scale_values


def train_isolation_forest(train_df, feature_cols):
    X_train, iso_cols, fill_values, scale_values = get_iso_forest_matrix(train_df, feature_cols)
    model = IsolationForest(**ISO_FOREST_PARAMS)
    model.fit(X_train)
    return model, iso_cols, fill_values, scale_values


def add_isolation_forest_scores(df, iso_model, iso_cols, fill_values, scale_values):
    df = df.copy()
    X, _, _, _ = get_iso_forest_matrix(df, iso_cols, fill_values=fill_values, scale_values=scale_values)
    X = X[iso_cols]
    df["iso_forest_score"] = (-iso_model.score_samples(X)).astype("float32")
    df["iso_forest_outlier_flag"] = (iso_model.predict(X) == -1).astype("int8")
    return df


def evaluate_isolation_forest_split(df, feature_cols, dataset_name):
    train_split, holdout_split = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL] if df[TARGET_COL].nunique() > 1 else None,
    )

    iso_model, iso_cols, iso_fill_values, iso_scale_values = train_isolation_forest(train_split, feature_cols)
    train_scored = add_isolation_forest_scores(train_split, iso_model, iso_cols, iso_fill_values, iso_scale_values)
    holdout_scored = add_isolation_forest_scores(holdout_split, iso_model, iso_cols, iso_fill_values, iso_scale_values)

    print(f"\nIsolation Forest fitted for {dataset_name} on {len(train_scored):,} training rows using columns: {iso_cols}")
    print(f"--- {dataset_name} Isolation Forest train/holdout check ---")
    for split_name, data_frame in [("train", train_scored), ("holdout", holdout_scored)]:
        if data_frame[TARGET_COL].nunique() > 1:
            auc = roc_auc_score(data_frame[TARGET_COL], data_frame["iso_forest_score"])
            ap = average_precision_score(data_frame[TARGET_COL], data_frame["iso_forest_score"])
            flagged_fraud_rate = data_frame.loc[data_frame["iso_forest_outlier_flag"] == 1, TARGET_COL].mean()
            print(
                f"  {split_name:10s} AUC={auc:.4f}  AUC-PR={ap:.4f}  "
                f"fraud-rate-among-flagged-outliers={flagged_fraud_rate:.4%}"
            )

    return iso_model, iso_cols, iso_fill_values, iso_scale_values


synthetic_iso_model, synthetic_iso_cols, synthetic_iso_fill_values, synthetic_iso_scale_values = evaluate_isolation_forest_split(
    train_df,
    SYNTHETIC_ISO_FOREST_COLS,
    "synthetic",
)
creditcard_iso_model, creditcard_iso_cols, creditcard_iso_fill_values, creditcard_iso_scale_values = evaluate_isolation_forest_split(
    valid_df,
    CREDITCARD_ISO_FOREST_COLS,
    "creditcard",
)

fraud_classifier, fraud_feature_names, fraud_feature_importances, fraud_feature_columns = train_fraud_classifier(train_df)

def seeded_rng(txn_id: str | None) -> random.Random:
    if txn_id:
        return random.Random(sum(ord(c) for c in txn_id))
    return random.Random(42)


def fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def pct(v: float) -> str:
    return f"{v*100:.1f}%"


def risk_from_prob(prob: float) -> tuple[str, str, str]:
    """Return (risk_level, color_label, recommended_action)"""
    if prob >= 0.80:
        return "Critical", "RED", "Block Transaction"
    if prob >= 0.65:
        return "High", "ORANGE", "Manual Review"
    if prob >= 0.45:
        return "Medium", "YELLOW", "Approve with Monitoring"
    return "Low", "GREEN", "Approve"


def synth_sample(txn_id: str | None = None) -> dict:
    rng = seeded_rng(txn_id)
    amount = round(rng.uniform(12.0, 12000.0), 2)
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Basic transaction info
    tx = {
        "transaction_id": txn_id or f"TXN-{int(time.time())}-{rng.randint(1000,9999)}",
        "timestamp": timestamp,
        "amount": amount,
        "merchant_name": rng.choice(["Travel+ Co.", "ElectroMart", "QuickBite", "SafeStore"]),
        "merchant_category": rng.choice(["Travel & Lodging", "Electronics", "Food & Beverage", "Retail"]),
        "payment_method": rng.choice(["Card Present (Chip)", "Card Not Present", "Mobile Wallet", "Wire"]),
        "customer_id": f"CUST-{rng.randint(1000,9999)}",
        "location": rng.choice(["Istanbul, TR", "New York, US", "Lagos, NG", "Mumbai, IN"]),
        "device_id": rng.choice(["DEV-4F1C-UNSEEN", "DEV-01A2-REC", "DEV-7B3D-REC"]),
    }

    return {"txn": tx}


def build_anomaly_input(tx: dict) -> pd.DataFrame:
    timestamp = pd.Timestamp(tx["timestamp"])
    hour = timestamp.hour + timestamp.minute / 60.0
    return pd.DataFrame([
        {
            "amount": float(tx["amount"]),
            "amount_log1p": math.log1p(max(float(tx["amount"]), 0.0)),
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            **{col: np.nan for col in CREDITCARD_PCA_COLS},
        }
    ])


def build_fraud_explanations(feature_names, importances, top_n=5):
    total = float(np.sum(importances)) or 1.0
    top_indices = np.argsort(importances)[::-1][:top_n]
    explanations = []
    for index in top_indices:
        explanations.append(
            {
                "feature": str(feature_names[index]),
                "contrib_pct": round((float(importances[index]) / total) * 100.0, 1),
            }
        )
    return explanations


def score_transaction_with_models(sample: dict) -> dict:
    tx = sample["txn"]
    start = time.perf_counter()

    anomaly_input = build_anomaly_input(tx)
    anomaly_start = time.perf_counter()
    anomaly_scored = add_isolation_forest_scores(
        anomaly_input,
        creditcard_iso_model,
        creditcard_iso_cols,
        creditcard_iso_fill_values,
        creditcard_iso_scale_values,
    )
    anomaly_ms = (time.perf_counter() - anomaly_start) * 1000.0
    iso_score = float(anomaly_scored["iso_forest_score"].iloc[0])
    iso_status = "Suspicious" if iso_score >= 0.65 else "Normal"

    fraud_input = build_fraud_model_row(tx)
    fraud_input_encoded = encode_fraud_features(fraud_input, fitted_columns=fraud_feature_columns)
    predict_start = time.perf_counter()
    fraud_prob = float(fraud_classifier.predict_proba(fraud_input_encoded)[0, 1])
    predict_ms = (time.perf_counter() - predict_start) * 1000.0
    prediction = "Fraudulent" if fraud_prob >= 0.5 else "Legitimate"
    confidence = max(fraud_prob, 1.0 - fraud_prob)

    explain_start = time.perf_counter()
    feature_highlights = build_fraud_explanations(fraud_feature_names, fraud_feature_importances)
    explain_ms = (time.perf_counter() - explain_start) * 1000.0

    total_ms = (time.perf_counter() - start) * 1000.0

    sample = dict(sample)
    sample["iso_score"] = iso_score
    sample["iso_status"] = iso_status
    sample["fraud_prob"] = fraud_prob
    sample["prediction"] = prediction
    sample["confidence"] = confidence
    sample["features"] = feature_highlights
    sample["times"] = {
        "anomaly_ms": round(anomaly_ms, 2),
        "predict_ms": round(predict_ms, 2),
        "explain_ms": round(explain_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    return sample


def printable_report(sample: dict) -> str:
    tx = sample["txn"]
    iso_score = sample["iso_score"]
    iso_status = sample["iso_status"]
    fraud_prob = sample["fraud_prob"]
    prediction = sample["prediction"]
    confidence = sample["confidence"]
    features = sample["features"]
    times = sample["times"]

    risk_level, color_label, action = risk_from_prob(fraud_prob)

    lines = []
    sep = "=" * 72
    lines.append(sep)
    lines.append("TRANSACTION ANALYSIS REPORT")
    lines.append(sep)
    lines.append("\n[1] Transaction Information")
    lines.append(f"  Transaction ID : {tx['transaction_id']}")
    lines.append(f"  Timestamp      : {tx['timestamp']}")
    lines.append(f"  Amount         : {fmt_money(tx['amount'])}")
    lines.append(f"  Merchant       : {tx['merchant_name']} ({tx['merchant_category']})")
    lines.append(f"  Payment Method : {tx['payment_method']}")
    lines.append(f"  Customer ID    : {tx['customer_id']}")
    lines.append(f"  Location       : {tx['location']}")
    lines.append(f"  Device ID      : {tx['device_id']}")

    lines.append('\n[2] Stage 1 — Anomaly Detection')
    lines.append(f"  IsolationForest score : {iso_score:.3f}")
    lines.append(f"  Anomaly status        : {iso_status}")

    lines.append('\n[3] Stage 2 — Fraud Classification')
    lines.append(f"  Prediction            : {prediction}")
    lines.append(f"  Fraud probability     : {pct(fraud_prob)}")
    lines.append(f"  Confidence score      : {pct(confidence)}")

    lines.append('\n[4] Risk Assessment')
    lines.append(f"  Risk level            : {risk_level} ({color_label})")
    lines.append(f"  Recommended action    : {action}")

    lines.append('\n[5] Explainability (top features by contribution)')
    for f in features:
        lines.append(f"  {f['feature']:25s} {f['contrib_pct']:5.1f}%  (global model importance)")

    lines.append('\n[6] Human-readable explanation')
    reasons = []
    if tx['device_id'].endswith('UNSEEN') or 'UNSEEN' in tx['device_id']:
        reasons.append('originates from an unseen device')
    if tx['amount'] > 2000:
        reasons.append('amount unusually high for this customer')
    if iso_status == 'Suspicious':
        reasons.append('anomaly score indicates unusual behavior compared to baseline')
    if len(reasons) == 0:
        hr = 'No strong risk signals; falls within expected patterns for this account.'
    else:
        hr = 'The transaction is high risk because it ' + ', '.join(reasons) + '.'
    lines.append(f"  {hr}")

    lines.append('\n[7] Performance Statistics')
    lines.append(f"  Total transaction time : {times['total_ms']} ms")
    lines.append(f"  Anomaly detection     : {times['anomaly_ms']} ms")
    lines.append(f"  Fraud model prediction : {times['predict_ms']} ms")
    lines.append(f"  Feature explanation    : {times['explain_ms']} ms")

    lines.append('\n[8] Decision Summary')
    lines.append(f"  Final prediction      : {prediction}")
    lines.append(f"  Fraud probability     : {pct(fraud_prob)}")
    lines.append(f"  Risk level            : {risk_level}")
    lines.append(f"  Recommended action    : {action}")
    lines.append(f"  Overall confidence    : {pct(confidence)}")

    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Transaction analysis CLI (prototype)')
    parser.add_argument('--id', '-i', help='Transaction ID to seed sample generation')
    args = parser.parse_args()

    start = time.perf_counter()
    sample = score_transaction_with_models(synth_sample(args.id))
    out = printable_report(sample)
    total_time = (time.perf_counter() - start) * 1000.0

    print(out)
    print(f"[note] transaction report wall time: {total_time:.1f} ms")


if __name__ == '__main__':
    main()
