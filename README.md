# Fraud Detection Data Pipeline & Models

A real-data credit card fraud detection project using human-readable features via **LightGBM** and **XGBoost**.

## Overview

This project aggregates multiple public fraud datasets into a single standardized 90k transaction CSV. It trains and evaluates two production-ready models based on human-readable location, merchant, and temporal features.

- Targets 35% fraud rate with 90,091 transactions.
- Chronological split using transaction date-times.
- Max-F1 threshold fitted strictly on validation split.
- Sub-millisecond latency. 

> **Status: research-grade proof of concept.**
> The models are trained on real tabular data formats to evaluate explainability (SHAP) and dual-model behavior.

## Dataset

- **Merged Dataset (`data/fraud_merged.csv`):** 90,091 transactions (35% fraud)
- **Sources:** Sparkov (`fraudTrain.csv`), BankSim, and FDH Handbook (legacy pickle)
- **Features:** 19 standardized human-readable attributes including `amount`, `category`, `merchant`, `age_bucket`, `lat`, `long`, `channel`, etc. (No PCA attributes).

*Note: The older ULB `creditcard.csv` (284k transactions, PCA features) has been removed from the core training process to focus on explainable real-world feature types.*

## Quickstart

Run via `.venv` with `pip install -r requirements.txt`.

### CLI Menu

An interactive CLI for training, evaluation, and latency tests:

```bash
.venv/Scripts/python.exe scripts/cli_menu.py
```

### Train (Pipeline)

Runs the full extraction, temporal split, training (LightGBM & XGBoost), evaluation, and benchmarking pipeline.

```bash
.venv/Scripts/python.exe scripts/run_pipeline.py
```

### Batch Scoring 

Predict custom CSVs or JSON payloads via CLI module:

```bash
.venv/Scripts/python.exe -m fraud_detection.predict in.csv --out out.csv
```

Pass `--explain` for per-transaction top-k SHAP local explainability.

## Output Structure 

- **Prediction:** `fraud_probability` (float) and `prediction` (0 or 1).
- **Explanation:** If requested, adds `explain_top_features` and `explain_top_shap_values`. 
- **Saved Artifacts:** `models/fraud_model.txt` (LightGBM), `models/fraud_model_xgboost.json`, and `<model>_threshold.txt`.

## Data Drift (PSI)

Includes a standalone monitor script for tracking feature drift via Population Stability Index on numeric metrics:
```bash
.venv/Scripts/python.exe -m fraud_detection.monitor --sample incoming.csv
```