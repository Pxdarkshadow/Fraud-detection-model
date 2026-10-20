"""Fraud scoring CLI + importable scorer (LightGBM + optional SHAP)."""

import os
import sys
import json
import argparse
import time

import pandas as pd
import numpy as np

import lightgbm as lgb
import joblib

from .config import (
    INPUT_COLS,
    FEATURES,
    CATEGORICAL_COLS,
    CLASS_COL,
    MODEL_PATH,
    THRESHOLD_PATH,
    ISO_FOREST_MODEL_PATH,
    ISO_FOREST_NUMERIC_COLS,
    RISK_LEVELS,
)


_shap_cache = {}
_iso_model_cache = None


def load_model(path=MODEL_PATH):
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return lgb.Booster(model_file=path)


def load_isolation_forest(path=ISO_FOREST_MODEL_PATH):
    global _iso_model_cache
    if _iso_model_cache is not None:
        return _iso_model_cache
    path = str(path)
    if os.path.isfile(path):
        try:
            _iso_model_cache = joblib.load(path)
            return _iso_model_cache
        except Exception:
            return None
    return None


def default_threshold():
    """models/threshold.txt (plain float) if present, else 0.5."""
    try:
        with open(THRESHOLD_PATH) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.5


def validate_input(df):
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # ponytail: drop global NaN check. Sparse logs have missing locations/demographics.


def engineer_features(df):
    df = df.copy()
    if CLASS_COL in df.columns:
        df.drop(columns=[CLASS_COL], inplace=True)

    if "amount" in df.columns:
        df["amount_log1p"] = np.log1p(pd.to_numeric(df["amount"], errors="coerce").clip(lower=0.0)).astype("float32")

    # Cyclical hour encoding
    if "hour" in df.columns:
        hour = pd.to_numeric(df["hour"], errors="coerce").fillna(12.0)
    elif "hour_of_day" in df.columns:
        hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(12.0)
    else:
        hour = pd.Series(12.0, index=df.index, dtype="float32")

    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")

    if all(x in df.columns for x in ["lat", "long", "merch_lat", "merch_long"]):
        dlat = df["lat"] - df["merch_lat"]
        dlon = (df["long"] - df["merch_long"]) * np.cos(np.radians(df["lat"].fillna(0)))
        df["distance_km"] = np.sqrt(dlat**2 + dlon**2) * 111.0
        df["distance_km"] = df["distance_km"].fillna(0.0)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Isolation Forest features if model is available
    iso_model = load_isolation_forest()
    if iso_model is not None:
        try:
            iso_cols = [c for c in ISO_FOREST_NUMERIC_COLS if c in df.columns]
            X = df[iso_cols].fillna(0.0).astype("float32")
            df["iso_forest_score"] = (-iso_model.score_samples(X)).astype("float32")
            df["iso_forest_outlier_flag"] = (iso_model.predict(X) == -1).astype("int8")
        except Exception:
            pass

    return df


def predict(df, threshold=None, model=None):
    if isinstance(df, (str, os.PathLike)):
        df = pd.read_csv(df) if str(df).endswith(".csv") else pd.read_json(df)
    validate_input(df)
    model = load_model() if model is None else model
    df_eng = engineer_features(df)
    threshold = default_threshold() if threshold is None else threshold

    feature_names = model.feature_name()
    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    probs = model.predict(df_eng[feature_names])
    df["fraud_probability"] = probs
    df["prediction"] = (probs >= threshold).astype(int)
    return df


def risk_from_prob(prob: float) -> tuple[str, str, str]:
    """Map fraud probability to (risk_level, color, recommended_action)."""
    for threshold, level, color, action in RISK_LEVELS:
        if prob >= threshold:
            return level, color, action
    return "Low", "GREEN", "Approve"


def _read_input(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict) and "columns" in raw and "data" in raw:
        return pd.DataFrame(raw["data"], columns=raw["columns"])
    return pd.DataFrame(raw)


def _explain_shap_top_k(model, X: pd.DataFrame, feature_cols, top_k=5):
    import shap

    key = (type(model).__name__, id(model), tuple(feature_cols), int(top_k))
    explainer = _shap_cache.get(key)
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        _shap_cache[key] = explainer

    shap_vals = explainer.shap_values(X, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    sv = np.asarray(shap_vals)
    abs_vals = np.abs(sv)

    k = min(top_k, abs_vals.shape[1])
    idx = np.argpartition(-abs_vals, kth=k - 1, axis=1)[:, :k]

    row_abs = np.take_along_axis(abs_vals, idx, axis=1)
    order = np.argsort(row_abs, axis=1)[:, ::-1]
    idx_sorted = np.take_along_axis(idx, order, axis=1)

    out = []
    for i in range(sv.shape[0]):
        feats = [feature_cols[j] for j in idx_sorted[i]]
        vals = [float(sv[i, j]) for j in idx_sorted[i]]
        out.append((feats, vals))
    return out


def explain_predictions(model, df: pd.DataFrame, top_k=5, feature_cols=None):
    feature_cols = feature_cols or model.feature_name()
    df_eng = engineer_features(df)
    for col in feature_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    t0 = time.perf_counter()
    _ = model.predict(df_eng[feature_cols])
    inference_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    explanations = _explain_shap_top_k(model, df_eng[feature_cols], feature_cols, top_k=top_k)
    explanation_ms = (time.perf_counter() - t0) * 1000

    top_features = []
    top_values = []
    for feats, vals in explanations:
        top_features.append(",".join(feats))
        top_values.append(",".join([f"{v:+.6f}" for v in vals]))

    return {
        "inference_ms": float(inference_ms),
        "explanation_ms": float(explanation_ms),
        "explain_top_features": top_features,
        "explain_top_shap_values": top_values,
    }


def main():
    p = argparse.ArgumentParser(description="Fraud prediction scorer")
    p.add_argument("input", nargs="?", help="CSV or JSON input file")
    p.add_argument("--in", dest="input_path", help="input file (alias for positional)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", default="out.csv")
    p.add_argument("--explain", action="store_true", help="Return top SHAP features")
    p.add_argument("--top-k", type=int, default=5, help="Top-K SHAP features")
    args = p.parse_args()

    path = args.input or args.input_path
    if not path:
        p.error("an input file is required")

    try:
        if path.lower().endswith(".json"):
            df = _read_input(path)
        else:
            df = pd.read_csv(path)

        validate_input(df)  # bad input exits here regardless of artifact

        model = load_model(args.model)
        df = engineer_features(df)

        threshold = args.threshold if args.threshold is not None else default_threshold()

        t0 = time.perf_counter()
        probs = model.predict(df[FEATURES])
        inference_ms = (time.perf_counter() - t0) * 1000

        df["fraud_probability"] = probs
        df["prediction"] = (probs >= threshold).astype(int)

        if args.explain:
            # ponytail: hard cap to keep SHAP under the <100ms target.
            if len(df) > 50:
                raise ValueError("--explain supports up to 50 rows to keep SHAP fast")
            expl = explain_predictions(model, df, top_k=args.top_k, feature_cols=FEATURES)
            df["explain_top_features"] = expl["explain_top_features"]
            df["explain_top_shap_values"] = expl["explain_top_shap_values"]
            print(f"Latency: inference={inference_ms:.2f}ms explanation={expl['explanation_ms']:.2f}ms")

        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} predictions to {args.out}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
