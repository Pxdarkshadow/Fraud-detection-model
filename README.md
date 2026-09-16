# Credit Card Fraud Detection — LightGBM

Binary classifier scoring a card transaction as **fraud / legitimate**, trained on the **ULB Credit
Card Fraud** dataset (real European card transactions, PCA-anonymized). Training pipeline
(`src/fraud_detection/pipeline.py`), a scoring entrypoint (`src/fraud_detection/predict.py`),
and plain-assert tests (`tests/test_predict.py`).
LightGBM, CPU inference, no external services.

> **Status: proof of concept, real data, still not production.** Numbers below are real and honest
> (unlike the earlier PaySim artifact), but the deployment pieces — monitoring, drift detection,
> versioning, an API, audit logging — are not built. See §7.

---

## 1. Quick answers

| Question | Answer |
|---|---|
| What does it do? | Predicts probability a card transaction is fraud. |
| Model type | LightGBM (gradient-boosted trees), binary logistic loss. |
| Input | `Time, V1…V28, Amount` (30 features, pre-derived by dataset owner, no labels). |
| Output | `fraud_probability` [0,1] + `prediction` at a threshold. Default threshold read from `models/threshold.txt` (fitted on validation only). |
| Train / Val / Test | 172,090 / 59,587 / 53,130 rows — chronological split by `Time`. |
| Dataset | ULB (real European card, 284,807 rows, 0.17% fraud, PCA-masked). |
| Test quality | AUC **0.899** · AUC-PR **0.462** · P/R/F1 **0.60 / 0.70 / 0.65** at val-fitted threshold. |
| Latency | Single-row **p99 0.35 ms**. |
| Model files | `models/fraud_model.txt` (~kB) + `models/threshold.txt`. |

**Honest caveat:** AUC 0.899 on real data is far below the earlier PaySim 0.998 — that gap is the
point. PaySim's near-perfect score came from a synthetic rule (fraud drains the account to zero);
this dataset has no such artifact. Expect deployment-time degradation from drift anyway.

### Known limitation — single-tree model

Early stopping fired at **iteration 1** (1 tree, threshold 1.0). The validation slice is small
(~61 frauds, 0.10%) so the AUC metric is noisy and stopped training early. The model works but
underfits: a tuned ULB model typically reaches ~0.97 AUC. Fixing it means a lower learning rate /
stratified split / longer patience, then re-running the pipeline. Deliberately not churned post-wiring.

---

## 2. Project structure

```
fraud-detection-model/
├── src/fraud_detection/     # Core package
│   ├── config.py            # Shared constants (paths, features, params)
│   ├── pipeline.py          # Training pipeline (load → split → train → evaluate → save)
│   └── predict.py           # Scoring CLI + importable scorer
├── tests/
│   └── test_predict.py      # Plain-assert tests
├── scripts/
│   ├── run_pipeline.py      # Training entrypoint
│   └── run_tests.py         # Test runner
├── data/                    # Datasets (creditcard.csv)
├── models/                  # Saved model + threshold
├── benchmarks/              # Latency/size benchmark results (JSON)
├── configs/
│   └── config.yaml          # Reference hyperparameters (code reads config.py)
├── reports/                 # Output reports
└── requirements.txt
```

## 3. Pipeline

```
data/creditcard.csv (284,807 rows, ULB real card data)
   |  load_data()
   v
engineer_features()  -> drops the Class label, nothing else (features are pre-derived)
   v
chronological split by "Time" (seconds since first tx) -> train / val / test (no shuffle)
   |  chrono_split(0.15 val, 0.15 test)
   v
LightGBM train, early stop on val AUC
   v
threshold = max-F1 fitted on VALIDATION ONLY  -> saved to models/threshold.txt
   v
evaluate val + test with that fixed threshold; feature importance; latency
   v
models/fraud_model.txt
```

Time-sorted split caveat: fraud in ULB is bursty and time-correlated, so split slices have uneven
fraud rates (train 0.21% / val 0.10% / test 0.13%). Test is the last 30% of the 2-day window — the
honest worst-case regime.

## 4. Commands

```bash
# Train (load -> split -> train -> threshold -> evaluate -> save model + threshold)
.venv\Scripts\python.exe scripts/run_pipeline.py

# Score transactions (CSV or JSON in, predictions out)
.venv\Scripts\python.exe -m fraud_detection.predict in.csv --out out.csv
.venv\Scripts\python.exe -m fraud_detection.predict features.json --out out.csv
#   JSON: list-of-objects, or {"columns":[...], "data":[[...],...]}
#   --threshold overrides models/threshold.txt; rejects missing/NaN features, exit 1
#   Run from src/ directory: cd src && python -m fraud_detection.predict ...

# Self-check (loads the saved model + dataset, plain asserts)
.venv\Scripts\python.exe scripts/run_tests.py
```

## 5. Results (run 2026-09-15, hardware Intel i7-13650HX, CPU)

| Metric | Validation | Test |
|---|---|---|
| AUC | 0.940 | **0.899** |
| AUC-PR | 0.388 | **0.462** |
| Threshold (val max-F1) | — | 1.0000 |
| Precision / Recall / F1 | 0.52 / 0.72 / 0.60 | **0.60 / 0.70 / 0.65** |
| Latency p99 | — | **0.35 ms** |

Top features (gain): V14, V4, V12, V8, V25.

## 6. Reproducibility

- Seed `42`; chronological split deterministic; threshold fitted on validation only, test evaluated
  once with the frozen threshold.
- `predict.py` and `test_predict.py` depend only on the feature contract
  (`Time, V1…V28, Amount`), `models/`, and `data/creditcard.csv`.

## 7. Dataset history

Training started on **PaySim** (`data/Fraud.csv`, still on disk, $6.36 M$ rows) — synthetic, and
its fraud rule (drain account to zero) produced a meaningless 0.998 AUC. Switched to **ULB** (real,
auth-free, 284,807 rows) for a harder, honest signal. Bundesbank and BAF were evaluated but rejected
(auth-gated / GitHub-LFS 404). No further datasets added.
