# Credit Card Fraud Detection — LightGBM

A real-data credit card fraud detection model built with **LightGBM**, designed for fast CPU inference and evaluated without synthetic data.

## Overview

This project:

- Trains on the real **ULB Credit Card Fraud** dataset.
- Uses a chronological train/validation/test split.
- Predicts the probability that a transaction is fraudulent.
- Fits the classification threshold on validation data only.
- Evaluates ranking, classification, model-size, and latency metrics.
- Supports CSV and JSON transaction scoring through the terminal.

> **Status: research-grade proof of concept.**  
> The model is trained and evaluated on real data, but this is not a production banking deployment. Production monitoring, drift detection, model registry/versioning, authentication, audit logging, and a production API are not included.

---

## 1. Model summary

| Item | Details |
|---|---|
| Task | Binary credit-card fraud classification |
| Model | LightGBM gradient-boosted decision trees |
| Objective | Binary logistic loss |
| Dataset | ULB Credit Card Fraud |
| Dataset size | 284,807 transactions |
| Fraud rate | Approximately 0.17% |
| Input features | `Time`, `V1`–`V28`, `Amount` |
| Number of features | 30 |
| Output | `fraud_probability` and threshold-based `prediction` |
| Training | CPU |
| Inference | CPU |
| Test ROC-AUC | **0.9899** |
| Test AUC-PR | **0.8006** |
| Test precision | **0.90** |
| Test recall | **0.75** |
| Test F1 | **0.82** |
| Single-row latency | **0.55 ms p99** |
| Model file | `models/fraud_model.txt` |
| Threshold file | `models/threshold.txt` |

### Why accuracy is not the main metric

The dataset is extremely imbalanced. A model can achieve very high accuracy while still missing fraudulent transactions.

The main metrics used here are:

- **ROC-AUC:** Overall ranking quality.
- **AUC-PR:** More informative under severe class imbalance.
- **Precision:** Percentage of flagged transactions that are fraud.
- **Recall:** Percentage of fraud transactions detected.
- **F1:** Balance between precision and recall.
- **p99 latency:** Latency at the 99th percentile of measured requests.

---

## 2. Project structure

```text
fraud-detection-model/
├── src/
│   └── fraud_detection/
│       ├── config.py            # Paths, features, and parameters
│       ├── pipeline.py          # Training and evaluation pipeline
│       └── predict.py           # CSV/JSON scoring CLI
├── tests/
│   └── test_predict.py          # Plain-assert tests
├── scripts/
│   ├── run_pipeline.py          # Training entrypoint
│   └── run_tests.py             # Test runner
├── data/
│   └── creditcard.csv           # Real ULB dataset
├── models/
│   ├── fraud_model.txt          # Saved LightGBM model
│   └── threshold.txt            # Validation-fitted threshold
├── benchmarks/                  # Latency and model-size results
├── configs/
│   └── config.yaml              # Reference configuration
├── reports/                     # Generated reports
├── requirements.txt
└── README.md
```

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

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Usage

### Train the model

Run from the project root:

```bash
.venv\Scripts\python.exe scripts/run_pipeline.py
```

The training pipeline performs:

1. Dataset loading and validation.
2. Feature preparation.
3. Chronological splitting.
4. LightGBM training with early stopping.
5. Validation threshold selection.
6. Validation and test evaluation.
7. Model and threshold saving.
8. Report and benchmark generation.

### Run tests

```bash
.venv\Scripts\python.exe scripts/run_tests.py
```

### Score a CSV file

```bash
.venv\Scripts\python.exe -m fraud_detection.predict in.csv --out out.csv
```

### Score a JSON file

```bash
.venv\Scripts\python.exe -m fraud_detection.predict features.json --out out.csv
```

Supported JSON formats include:

```json
[
  {
    "Time": 0,
    "V1": 1.2,
    "V2": -0.4,
    "V3": 0.8,
    "Amount": 12.5
  }
]
```

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
