"""Fraud scoring CLI + importable scorer (LightGBM or XGBoost + optional SHAP)."""

import os
import sys
import json
import argparse
import time

import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb

from .config import (
    INPUT_COLS,
    FEATURES,
    CATEGORICAL_COLS,
    CLASS_COL,
    MODEL_PATH,
    THRESHOLD_PATH,
    XGB_MODEL_PATH,
    XGB_THRESHOLD_PATH,
)


_shap_cache = {}


def _is_xgb_model_path(path: str) -> bool:
    # ponytail: heuristic based on extension; upgrade path: store a small model metadata file.
    p = str(path).lower()
    return p.endswith(".json") or "xgboost" in p


def load_model(path=MODEL_PATH):
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")

    if _is_xgb_model_path(path):
        model = xgb.Booster()
        model.load_model(path)
        return model

    return lgb.Booster(model_file=path)


def default_threshold_for_model(model_path):
    model_path = str(model_path)
    if _is_xgb_model_path(model_path):
        try:
            with open(XGB_THRESHOLD_PATH) as f:
                return float(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0.5

    return default_threshold()


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
        df["amount_log1p"] = np.log1p(df["amount"])

    if all(x in df.columns for x in ["lat", "long", "merch_lat", "merch_long"]):
        dlat = df["lat"] - df["merch_lat"]
        dlon = (df["long"] - df["merch_long"]) * np.cos(np.radians(df["lat"].fillna(0)))
        df["distance_km"] = np.sqrt(dlat**2 + dlon**2) * 111.0
        df["distance_km"] = df["distance_km"].fillna(0.0)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _xgb_encode(X, feature_cols):
    """Encode categories to integers for XGBoost; returns a new DataFrame slice."""
    X = X[feature_cols].copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype) or str(X[c].dtype) == "category":
            X[c] = X[c].cat.codes
    return X


def predict(df, threshold=None, model=None):
    if isinstance(df, (str, os.PathLike)):
        df = pd.read_csv(df) if str(df).endswith(".csv") else pd.read_json(df)
    validate_input(df)
    model = load_model() if model is None else model
    df = engineer_features(df)
    threshold = default_threshold() if threshold is None else threshold

    if isinstance(model, xgb.Booster):
        probs = model.predict(xgb.DMatrix(_xgb_encode(df, FEATURES)))
    else:
        probs = model.predict(df[FEATURES])

    df["fraud_probability"] = probs
    df["prediction"] = (probs >= threshold).astype(int)
    return df


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
    feature_cols = feature_cols or FEATURES
    df = engineer_features(df)

    t0 = time.perf_counter()
    if isinstance(model, xgb.Booster):
        _ = model.predict(xgb.DMatrix(_xgb_encode(df, feature_cols)))
        X_shap = _xgb_encode(df, feature_cols)
    else:
        _ = model.predict(df[feature_cols])
        X_shap = df[feature_cols]
    inference_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    explanations = _explain_shap_top_k(model, X_shap, feature_cols, top_k=top_k)
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

        threshold = (
            args.threshold
            if args.threshold is not None
            else default_threshold_for_model(args.model)
        )

        # Inference (timed separately when --explain is enabled)
        t0 = time.perf_counter()
        if isinstance(model, xgb.Booster):
            probs = model.predict(xgb.DMatrix(_xgb_encode(df, FEATURES)))
        else:
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
