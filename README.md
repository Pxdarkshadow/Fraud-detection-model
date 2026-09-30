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

<<<<<<< HEAD
## Setup
=======
---

## 3. Dataset and features

The Model uses the real ULB Credit Card Fraud dataset:

- 284,807 real European card transactions.
- PCA-anonymized features.
- Severe class imbalance.
- No synthetic fraud generation in the reported training/evaluation run.
- 
The model expects these 30 columns:

```text
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

The original `Class` column is the target label and is not used as an inference feature.

The `V1`–`V28` columns are PCA-transformed/anonymized features supplied by the dataset. They do not directly represent human-readable attributes such as merchant category, customer age, device type, or location.

### Input validation

- All 30 feature columns must be present.
- Feature names must match the feature contract.
- Missing and NaN values are rejected by the scoring entrypoint.
- CSV and supported JSON input formats are accepted.

---

## 4. ML pipeline

```text
Load data
   │
   ▼
Validate dataset and feature contract
   │
   ▼
Remove target label from model inputs
   │
   ▼
Chronological train / validation / test split
   │
   ▼
Train LightGBM with early stopping
   │
   ▼
Fit max-F1 threshold on validation data only
   │
   ▼
Evaluate validation and test data
   │
   ▼
Save model, threshold, reports, and benchmarks
```

### Data split

| Split | Rows |
|---|---:|
| Train | 191,452 |
| Validation | 40,225 |
| Test | 53,130 |

The split is chronological by `Time`.

The validation region uses a stratified holdout within its chronological band to make early stopping less noisy while retaining the overall time-based evaluation design.

The test set remains separate and is evaluated using the threshold selected from validation data.

### Training configuration

```text
learning_rate       = 0.02
num_leaves          = 96
min_child_samples   = 50
max_boost_rounds    = 3000
early_stopping      = 300 rounds
scale_pos_weight    = 10
random_seed         = 42
best_iteration      = 773
```

The decision threshold is selected using validation F1 and then frozen before test evaluation.

---

## 5. Results

Results from the run on **2026-09-16** using an Intel Core i7-13650HX CPU:

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0.9859 | **0.9899** |
| AUC-PR | 0.8218 | **0.8006** |
| Threshold | — | **0.6267** |
| Precision | 0.85 | **0.90** |
| Recall | 0.80 | **0.75** |
| F1 | 0.82 | **0.82** |
| Latency p99 | — | **0.55 ms** |
| Best iteration / trees | — | **773 / 773** |

Top features by gain:

```text
V14, V10, V12, V17, V25
```

### Latency requirement

The recorded single-row p99 inference latency is **0.55 ms**, which is below the project's sub-100-ms target.

This is a local model-inference benchmark. It does not represent complete end-to-end production latency, which would also include network overhead, feature retrieval, validation, logging, queueing, and service overhead.

---

## 6. Installation

Create and activate a virtual environment:
>>>>>>> e5411d5938699f3f7c41806399b9973b06ee87df

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

<<<<<<< HEAD
One gradient-boosted tree model. No sequence model, no online learning, no serving layer. `distance_km` is a local flat-earth approximation, not haversine. Metrics reflect this merged public dataset, not a live card portfolio.
=======
or:

```json
{
  "columns": ["Time", "V1", "V2", "V3", "Amount"],
  "data": [[0, 1.2, -0.4, 0.8, 12.5]]
}
```

The actual input must contain all 30 required features.

### Override the threshold

```bash
.venv\Scripts\python.exe -m fraud_detection.predict in.csv --out out.csv --threshold 0.70
```

The scorer rejects missing or NaN features and exits with code `1` when input validation fails.

If running directly from the `src` directory:

```bash
cd src
python -m fraud_detection.predict ...
```

---

## 8. Output format

The scorer returns the original transaction fields with two additional columns:

| Column | Meaning |
|---|---|
| `fraud_probability` | Model-estimated fraud score in `[0, 1]` |
| `prediction` | `1` for fraud and `0` for legitimate according to the selected threshold |

A probability is a model score, not a guarantee that a transaction is fraudulent.

---

## 9. Reproducibility and leakage controls

- Random seed: `42`.
- The split is deterministic.
- The threshold is fitted on validation data only.
- The test set is evaluated using the frozen validation threshold.
- The target column `Class` is excluded from inference features.
- No synthetic transactions are used for the reported ULB results.
- The reported metrics should be regenerated whenever code, dependencies, data, feature handling, or configuration changes.

---
>>>>>>> e5411d5938699f3f7c41806399b9973b06ee87df
