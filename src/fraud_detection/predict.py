"""Fraud scoring CLI + importable scorer (ULB credit-card, LightGBM)."""

import os
import sys
import json
import argparse

import pandas as pd
import lightgbm as lgb

from .config import INPUT_COLS, CLASS_COL, MODEL_PATH, THRESHOLD_PATH


def load_model(path=MODEL_PATH):
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return lgb.Booster(model_file=path)


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
    nulls = [c for c in INPUT_COLS if df[c].isna().any()]
    if nulls:
        raise ValueError(f"NaN found in required columns: {nulls}")


def engineer_features(df):
    df = df.copy()
    if CLASS_COL in df.columns:
        df.drop(columns=[CLASS_COL], inplace=True)
    return df


def predict(df, threshold=None):
    if isinstance(df, (str, os.PathLike)):
        df = pd.read_csv(df) if str(df).endswith(".csv") else pd.read_json(df)
    validate_input(df)
    model = load_model()
    df = engineer_features(df)
    threshold = default_threshold() if threshold is None else threshold
    probs = model.predict(df[INPUT_COLS])
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


def main():
    p = argparse.ArgumentParser(description="Fraud prediction scorer")
    p.add_argument("input", nargs="?", help="CSV or JSON input file")
    p.add_argument("--in", dest="input_path", help="input file (alias for positional)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", default="out.csv")
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
        probs = model.predict(df[INPUT_COLS])
        df["fraud_probability"] = probs
        df["prediction"] = (probs >= threshold).astype(int)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} predictions to {args.out}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
