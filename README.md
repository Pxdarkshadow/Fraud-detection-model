# Fraud Detection Model

A tabular fraud classifier. One LightGBM model scores transactions from human-readable features (amount, merchant, location, time, demographics). No web UI. Training and batch scoring are command-line only.

The dataset is a merged CSV of public fraud sources, rebalanced to about 35% fraud. The split is chronological on `tx_datetime`. The decision threshold is the F1 maximum on the validation slice and is never fitted on the test slice.

## Dataset

`data/fraud_merged.csv` — 90,091 transactions, 35% fraud.

Sources: Sparkov (`fraudTrain.csv`), BankSim, and the FDH Handbook extract. Columns:

`transaction_id`, `tx_datetime`, `amount`, `category`, `merchant`, `gender`, `age_bucket`, `city`, `state`, `lat`, `long`, `merch_lat`, `merch_long`, `channel`, `hour`, `day_of_week`, `is_weekend`, `source`, `is_fraud`.

The model also uses two derived columns: `amount_log1p` and a naive euclidean `distance_km` (degrees times 111). Categorical columns are passed to LightGBM as native categories.

## Project layout

```text
src/fraud_detection/
  config.py       paths, feature lists, split ratios
  pipeline.py     load, features, chronological split, train, evaluate, save
  predict.py      batch scoring (import or CLI)
  monitor.py      PSI drift on numeric features
scripts/
  merge_datasets.py   build fraud_merged.csv
  run_pipeline.py     train
  run_tests.py        tests
tests/test_predict.py
configs/config.yaml    reference only; code reads config.py
models/                 fraud_model.txt and threshold.txt after training
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

On macOS or Linux, the interpreter is `.venv/bin/python`.

## Train

```bash
.venv/Scripts/python.exe scripts/run_pipeline.py
```

Reads `data/fraud_merged.csv`. Sorts by `tx_datetime`. Newest 15% is the test set. The next 30% is a validation band; half of that band is a stratified holdout used only to fit the threshold, and the other half joins training. LightGBM trains with early stopping on validation AUC. Writes:

- `models/fraud_model.txt`
- `models/threshold.txt`

Also prints AUC, AUC-PR, precision, recall, F1, a confusion matrix, top features by gain, single-row latency, and a SHAP example.

## Score

Input CSV or JSON must include every raw column in `INPUT_COLS` (the dataset columns except `transaction_id`, `tx_datetime`, and `is_fraud`). Missing columns raise. Missing values in location or demographics are allowed.

```bash
.venv/Scripts/python.exe -m fraud_detection.predict data/sample.csv --out out.csv
```

Output adds `fraud_probability` and `prediction` (1 when the probability is at or above the saved threshold). `--explain` adds top-k SHAP columns. Explanations are limited to 50 rows.

From Python:

```python
from fraud_detection.predict import predict

scored = predict("data/sample.csv")
```

Run this from the repo root with `src` on `PYTHONPATH`, or install the package in editable mode. The training script inserts `src` itself.

## Drift

Population Stability Index on numeric features, reference versus a new sample:

```bash
.venv/Scripts/python.exe -m fraud_detection.monitor --reference data/fraud_merged.csv --sample incoming.csv
```

## Tests

```bash
.venv/Scripts/python.exe tests/test_predict.py
```

Plain asserts. No pytest required.

## Limits

One gradient-boosted tree model. No sequence model, no online learning, no serving layer. `distance_km` is a local flat-earth approximation, not haversine. Metrics reflect this merged public dataset, not a live card portfolio.
