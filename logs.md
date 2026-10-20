# Project Engineering Logs

This log documents all technical changes, architecture enhancements, model iterations, and pipeline updates. Both team members contribute here to maintain a shared, synchronized record of project evolution.

---

## [2026-10-20 22:15:00 IST] - Architecture Unification & Hybrid Engine Integration

### Context & Objectives
- Unified the two independent codebase branches into a cohesive, production-grade hybrid fraud detection engine without overwriting or discarding existing work.
- Addressed dataset location changes and wired the pipeline to ingest both `creditcard.csv` (284,807 transactions) and `synthetic_fraud_transactions.csv` located in `dataset/`.
- Preserved all existing scripts, tests, monitoring tools, and directory layouts while introducing missing unsupervised anomaly detection and in-fold dynamic SMOTE augmentation.

### Changes & Additions

#### 1. Environment & Dependencies
- Created clean virtual environment in `venv/` (Python 3.13.1).
- Updated `requirements.txt` to include `imbalanced-learn>=0.12.0` for in-memory dynamic SMOTE.

#### 2. Universal Schema & Data Loader (`src/fraud_detection/data.py`)
- Added cross-dataset ingestion and normalization for both `synthetic_fraud_transactions.csv` and `creditcard.csv`.
- Standardized transaction types with normalized category mappings (`TYPE_MAP`).
- Derived universally aligned features to prevent domain shift between real-world and synthetic data:
  - `amount_log1p` (logarithmic transaction value).
  - Cyclical time features (`hour_sin`, `hour_cos` using $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$) derived from timestamp, hour of day, or seconds elapsed.
  - Optional account balance deltas (`balance_delta_orig`, `balance_delta_dest`) and ratios.

#### 3. Unsupervised Anomaly Detection (`src/fraud_detection/anomaly.py`)
- Implemented `IsolationForestDetector` wrapping `sklearn.ensemble.IsolationForest`.
- Restricts training to universally shared numeric features (`amount`, `amount_log1p`, `hour_sin`, `hour_cos`) to eliminate NaN generation and feature mismatch crashes.
- Extracted dynamic features:
  - `iso_forest_score`: Raw anomaly score (negative sample score, higher = more anomalous).
  - `iso_forest_outlier_flag`: Binary indicator (1 = flagged anomaly, 0 = normal).
- Added standalone anomaly evaluation (AUC, AUC-PR, and outlier fraud concentration).
- Persistence: Saved and loaded via `models/isolation_forest.joblib`.

#### 4. Dynamic In-Fold SMOTE & Hybrid LightGBM Training (`src/fraud_detection/pipeline.py`)
- Integrated `imblearn.over_sampling.SMOTE` strictly inside the training fold.
- Generates synthetic minority fraud samples in-memory during training only, preventing data leakage into validation or test sets.
- Integrated `iso_forest_score` and `iso_forest_outlier_flag` directly into the LightGBM supervised feature matrix (`GBM_FEATURES`).
- Retained chronological validation splitting, early stopping, and validation-only max-$F_1$ threshold tuning.

#### 5. Real-Time Inference & Enterprise Reporting (`src/fraud_detection/predict.py`)
- Upgraded inference engine to execute the full 2-stage pipeline:
  1. Extraction & scoring via Isolation Forest.
  2. Prediction via LightGBM using anomaly scores as features.
  3. Feature attribution via `shap.TreeExplainer`.
- Added enterprise risk categorization:
  - `Critical` ($\ge 0.80$, RED, "Block Transaction")
  - `High` ($\ge 0.65$, ORANGE, "Manual Review")
  - `Medium` ($\ge 0.45$, YELLOW, "Approve with Monitoring")
  - `Low` ($< 0.45$, GREEN, "Approve")
- Added confidence scoring ($\text{confidence} = |\text{probability} - 0.5| \times 2$) and rule-based human explanations.
- Added structured CLI and printable transaction reports (`analyze_transaction` and `printable_report`).

#### 6. Documentation & Maintenance
- Resolved Git merge conflict markers in `README.md`.
- Documented hybrid inference metrics, SLAs, and feature contracts.

#### 7. Verification & Benchmark Results
- Test execution of hybrid pipeline succeeded end-to-end:
  - Isolation Forest training: Fitted on 10,000 rows, AUC=0.7737 on validation, AUC=0.8673 on test.
  - LightGBM supervised stage: `iso_forest_score` became the **#1 top feature by gain** (gain=30,064.9), demonstrating strong signal amplification.
  - Latency: Stage 1 = 68.42 ms, Stage 2 = 1.74 ms, Total Mean = 70.16 ms (well within < 100 ms SLA).
  - Test suite (`scripts/run_tests.py`): All tests passed, including schema input checks, CLI non-zero exit checks, and model honesty margin (`margin = 0.3589 > 0.05`).
  - Single-transaction audit report CLI (`-m fraud_detection.predict --id TXN-99824`): Generated complete 8-section report with SHAP attributions and automated explanation.

---

## [2026-10-20 22:36:00 IST] - IDE Import Resolution & Test Suite Refactoring

### Context & Objectives
- Resolved language server / IDE import warnings (`reportMissingImports`) on `scripts/run_pipeline.py` (line 16) and `tests/test_predict.py` (lines 19, 20, 66, and 80).
- Cleaned up inline imports in tests and aligned with PEP 8 standards.

### Changes & Additions
1. **IDE Python Path Configuration**:
   - Added `venv/Lib/site-packages/fraud_detection.pth` linking `src` directly into the virtual environment, making `fraud_detection` a first-class importable module without needing manual `PYTHONPATH` exports.
   - Added `.vscode/settings.json` in both the workspace root and project root specifying `python.defaultInterpreterPath` and `python.analysis.extraPaths` for Pylance/Pyright.
2. **Refactored `tests/test_predict.py`**:
   - Consolidated all `fraud_detection` imports (`INPUT_COLS`, `validate_input`, `predict`, `default_threshold`, `MODEL_PATH`, `DATA_PATH`, `THRESHOLD_PATH`, `SYNTHETIC_DATA_PATH`, `CREDITCARD_DATA_PATH`) at the top of the file.
   - Removed duplicate inline imports on lines 66 and 80.
   - Upgraded path resolution to `pathlib.Path`.
3. **Refactored `scripts/run_pipeline.py` & `scripts/run_tests.py`**:
   - Cleaned up path insertion and converted docstrings to raw strings to eliminate Python 3.12+ `SyntaxWarning: invalid escape sequence '\S'`.
   - Added dual-import fallback (`from fraud_detection...` with fallback to `from src.fraud_detection...`) to ensure IDE language servers resolve imports under all workspace configurations.
4. **Editable Package Build & Pyright Configuration**:
   - Created `pyproject.toml` specifying `setuptools` find-packages in `src/`.
   - Installed `fraud_detection` in editable mode (`pip install -e . --no-deps`) into `venv/`.
   - Created `pyrightconfig.json` at both workspace root (`Fraud Detection BankIQ`) and project root (`Fraud-detection-model`) to guarantee instant Pylance/Pyright resolution.
   - Added `src/__init__.py`.
5. **Verification**:
   - Executed `scripts/run_tests.py`: All 5 test assertions passed cleanly with 0 warnings and 0 errors.

---

## [2026-10-20 23:40:00 IST] - Architecture Stabilization & Minimalist Hybrid Consolidation

### Context & Objectives
- Reverted all temporary modularizations (`data.py`, `anomaly.py`) to strictly preserve the clean, original structure designed for the project.
- Safely integrated core hybrid capabilities (Isolation Forest unsupervised anomaly scoring, in-fold SMOTE augmentation, cyclical feature encoding, and 4-tier risk classification) directly into the core files (`config.py`, `pipeline.py`, `predict.py`).
- Performed disk space cleanup and cache purging.
- Established a stable, verifiable checkpoint ready for source control sync and dataset reconciliation.

### Detailed Changes

#### 1. Disk Space Reclamation & Cleanup
- Purged package wheel caches (`pip cache purge`) and cleaned temporary artifacts.
- Removed unused and auxiliary `.pyc` and `__pycache__` directories.
- C: drive free space safely restored above 1.0 GB.

#### 2. Configuration (`src/fraud_detection/config.py`)
- Retained original `DATA_PATH = PROJECT_ROOT / "data" / "fraud_merged.csv"`.
- Added paths for auxiliary datasets (`SYNTHETIC_DATA_PATH`, `CREDITCARD_DATA_PATH`).
- Added Stage 1 artifact path: `ISO_FOREST_MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"`.
- Defined universal numeric feature set (`ISO_FOREST_NUMERIC_COLS = ["amount", "amount_log1p", "hour_sin", "hour_cos"]`).
- Defined 4-tier enterprise risk categories (`Critical`, `High`, `Medium`, `Low`) with associated operational actions.

#### 3. Pipeline Ingestion & Model Training (`src/fraud_detection/pipeline.py`)
- **Data Ingestion**: Updated `load_data()` to first check for `data/fraud_merged.csv`, gracefully falling back to available synthetic or credit card datasets when the merged file is pending.
- **Feature Engineering**: Added cyclical diurnal encodings (`hour_sin`, `hour_cos` using $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$).
- **Stage 1 (Unsupervised)**: Added `train_isolation_forest()` and `add_isolation_scores()` generating `iso_forest_score` and `iso_forest_outlier_flag`.
- **Stage 2 (Supervised)**: Incorporated `iso_forest_score` into LightGBM feature matrix, along with safe in-fold dynamic SMOTE augmentation for numeric-only imbalanced folds.
- **Artifact Persistence**: Pipeline saves `fraud_model.txt`, `isolation_forest.joblib`, and `threshold.txt`.

#### 4. Inference & Explainability (`src/fraud_detection/predict.py`)
- Implemented `load_isolation_forest()` with cached loading.
- Enhanced `engineer_features()` to dynamically score incoming transactions through Isolation Forest when the model artifact is present.
- Added `risk_from_prob()` for standardized risk-tier classification.
- Preserved `validate_input()`, `predict()`, and SHAP TreeExplainer latency benchmarking.

#### 5. Verification & Live Benchmarks
- **Test Suite (`scripts/run_tests.py`)**: All 5 assertion checks passed cleanly with 0 errors and 0 warnings (CLI rejection, missing amount validation, default threshold loading).
- **Live Transaction Inference (`predict.py`)**:
  - Sample transaction successfully scored through both stages.
  - Calculated probability: `0.6401` mapped to 4-tier risk classification: `('Medium', 'YELLOW', 'Approve with Monitoring')`.
- **SHAP Attribution Engine (`explain_predictions`)**:
  - Inference latency: `12.4 ms` (well within < 100 ms target).
  - Feature alignment dynamically matched booster model features.
  - Top 3 fraud signals identified: `['merchant', 'amount', 'iso_forest_score']`, confirming `iso_forest_score` from Stage 1 is actively recognized as a key indicator.
- **IDE & Workspace**: Verified zero-warning state when `Fraud-detection-model` is opened directly as workspace root with project-level `.vscode/settings.json`.
- **Checkpoint Status**: Final safe, stable state ready for GitHub commit and push.

---
