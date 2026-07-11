# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, config-driven hyperparameter sweeps, three-way train/val/test split,
early stopping, adaptive learning rate scheduling, gradient clipping,
MLflow experiment tracking, containerised serving, structured logging,
input validation, and automated tests.

> **Status:** Phase 1 complete (`v1.0.0`). Phase 2 active on `dev` —
> MLflow tracking is wired in; full hyperparameter search with Optuna,
> walk-forward validation, and MLflow Model Registry are next.

---

## Project structure

```
lstm-time-series-mlops/
├── src/
│   ├── config.py                  # Single source of truth for all constants and defaults
│   ├── data/
│   │   ├── loader.py              # StockLoader — yfinance fetch + validation
│   │   └── preprocessor.py        # TimeSeriesPreprocessor — scaling, windowing, three-way split
│   ├── models/
│   │   └── lstm_model.py          # LSTMForecaster — architecture only
│   ├── training/
│   │   └── trainer.py             # TimeSeriesTraining — grid search, callbacks, MLflow logging
│   └── inference/
│       └── predictor.py           # Predictor — load model/scaler, predict, inverse-transform
├── app/
│   ├── main.py                    # FastAPI application
│   └── core/
│       └── logger.py              # Structured JSON logger
├── tests/
│   ├── test_preprocessor.py       # Windowing boundary cases + scaler round-trip
│   └── test_predictor.py          # Input validation (ValueError on wrong length)
├── notebooks/
│   ├── exploration.ipynb
│   └── train_legacy.py            # Original flat training script, kept for reference
├── models/                        # Gitignored — populated by training runs
├── mlruns/                        # Gitignored — MLflow local run metadata
├── mlartifacts/                   # Gitignored — MLflow model and scaler artifacts
├── mlflow.db                      # Gitignored — MLflow SQLite backend
├── Dockerfile
├── requirements.txt               # Top-level production dependencies
├── requirements.lock              # Full pinned environment (used by Docker)
├── requirements-dev.txt           # Dev-only dependencies (pytest)
└── README.md
```

---

## Model

| Property | Value |
|---|---|
| Architecture | Input(30,1) → LSTM(64) → LSTM(64) → Dense(128, relu) → Dropout(0.5) → Dense(1) |
| Input | 30-day rolling window of adjusted closing price |
| Target | Next-day closing price |
| Split | 70% train / 15% validation / 15% test, chronological — no shuffling |
| Scaler | `StandardScaler` fit on training data only (no leakage into val or test) |
| Optimizer | Adam with `clipnorm=1.0` (gradient clipping for LSTM stability) |
| Callbacks | `EarlyStopping` (patience=10, restore\_best\_weights) + `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Hyperparameter search | Grid search via `itertools.product` over `config.HYPER_PARAMS` |
| Model selection criterion | Validation RMSE% — test set never touched during selection |
| Experiment tracking | MLflow with SQLite backend and dedicated artifact store |

---

## Results

| Split | Period | RMSE% | Notes |
|---|---|---|---|
| Validation | 2021–2024 | ~3–4% | Used for model selection |
| Test | 2024–2026 | ~14–17% | Held-out, touched once after selection |

The gap between validation and test RMSE% is primarily attributable to
**distribution shift**: the model was trained on 2010–2021 price data, while
the test period (2024–2026) represents a structurally different price regime.
The `StandardScaler` was fit on training data only, so test-period prices
fall partially outside the scaler's fitted distribution.

These results reflect a **baseline configuration** — only `batch_size` was
swept during hyperparameter search, with all other parameters held at
defaults. A full Optuna-based sweep is scoped for Phase 2.

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow 2.18.0 / Keras 3.15.0 |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn `StandardScaler` |
| Experiment tracking | MLflow 3.14.0 (SQLite backend) |
| Serving | FastAPI + Uvicorn |
| Persistence | joblib (scaler), Keras native `.keras` format (model) |
| Containerisation | Docker (python:3.12-slim) |
| Tests | pytest |

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/satar-shams/lstm-time-series-mlops.git
cd lstm-time-series-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

### 2. Start the MLflow server

Run this in a **separate terminal** before training. MLflow must be running
for experiment tracking to work.

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:./mlartifacts
```

The MLflow UI will be available at `http://127.0.0.1:5000`.

### 3. Train the model

```bash
python -m src.training.trainer
```

Fetches AAPL data via yfinance, runs a hyperparameter grid search with early
stopping and adaptive learning rate scheduling, logs every config as a
separate MLflow run (params, metrics, model artifact for the winner),
evaluates on the held-out test set, then saves the best model locally to
`models/best_model.keras` and the fitted scaler to `models/scaler.bin`.

### 4. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Test the endpoints

```bash
curl http://localhost:8000/
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      276.58, 283.92, 287.25, 287.18, 293.05,
      292.68, 294.80, 298.87, 298.21, 300.23,
      297.84, 298.97, 302.25, 304.99, 308.82,
      308.33, 310.85, 312.51, 312.06, 306.31,
      315.20, 310.26, 311.23, 307.34, 301.54,
      290.55, 291.58, 295.63, 291.13, 296.42
    ]
  }'
```

Expected response:

```json
{"prediction": 293.33}
```

Input must contain exactly 30 float values (one per trading day).
Sending the wrong number returns a clear `422` error:

```json
{"detail": "Expected 30 values, got 29"}
```

---

## Run with Docker

```bash
docker build -t lstm-api .
docker run -p 8000:8000 lstm-api
```

> **Note:** The Docker image does not include trained model artifacts
> (`models/` is gitignored). Run the trainer locally first, then rebuild
> the image so `COPY models ./models` has real files to include.

---

## Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests cover:
- `create_windows` boundary cases (first, middle, last window) with
  hand-verified expected arrays
- Scaler fit/transform/inverse\_transform round-trip correctness
- `Predictor` input length validation (`ValueError` on wrong-length input)

---

## Configuration

All constants and default parameter values live in `src/config.py`.
`DEFAULTS_PARAMS` is the single source of truth for all model and training
defaults. `HYPER_PARAMS` defines which parameters are actively swept —
comment/uncomment entries to control the search space.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.85

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction"

DEFAULTS_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "lstm_units": 64,
    "dense_units": 128,
    "dropout_rate": 0.5,
    "clip_norm": 1.0,
}

HYPER_PARAMS = {
    "batch_size": [32, 64],
    # "learning_rate": [0.01, 0.001, 0.0001],
    # "lstm_units": [32, 64, 128],
    # "clip_norm": [0.5, 1.0, 2.0],
}
```

---

## Incident record: Keras/TensorFlow version drift

### Symptom

Model saved successfully but crashed at container startup:

```
TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

### Root cause

`tensorflow==2.18.0` does not pin a specific Keras version. Since TensorFlow
2.16, Keras 3 ships as an independently versioned package. The model was
trained with `keras==3.15.0`, but the Docker image — built from a
`requirements.txt` that only pinned `tensorflow` — resolved a different
Keras 3.x release at build time. The newer `Dense` layer config key
(`quantization_config`) was not recognised by the older version's
`Dense.__init__()`, breaking model deserialisation.

The CUDA warnings visible in container logs are unrelated — expected
CPU-fallback behaviour with no GPU passthrough.

### Fix

- Pinned `keras==3.15.0` explicitly alongside `tensorflow==2.18.0`
- Generated `requirements.lock` via `pip freeze` to capture the full
  resolved environment, not just top-level packages
- Rebuilt the Docker image from `requirements.lock` (not the loose file)
- Matched the Dockerfile's Python base image (`3.12-slim`) to the training
  venv's Python version

### Lesson

Pinning a top-level framework does not pin its sub-dependencies once that
framework decouples its own versioning. Train/serve environment parity must
be enforced at the full dependency-tree level via a lockfile.

---

## Known limitations

| Limitation | Notes |
|---|---|
| **Baseline hyperparameter search only** | Only `batch_size` swept in current implementation. Full Optuna-based sweep is Phase 2 scope. |
| **Distribution shift on test set** | Model trained on 2010–2021 data performs significantly worse on 2024–2026 test data due to price regime change. Walk-forward validation and returns-based modelling are planned mitigations. |
| **No retrain-on-90% workflow** | Standard practice is to retrain the best config on train+val combined after selection. Scoped for Phase 2. |
| **MLflow Model Registry not yet used** | Model artifacts are logged per run but not registered in the MLflow Model Registry. Registration and versioned promotion workflow is Phase 2 scope. |
| **Predictor test requires trained artifacts** | `tests/test_predictor.py` loads real model and scaler files from `models/`, which are gitignored. Proper fix is mocking `load_model`/`joblib.load`; deferred for now. |
| **No multi-step forecasting** | The model predicts one day ahead. Multi-day forecasting via recursive window-sliding is a planned `Predictor` extension. |

---

## Phase 2 roadmap

### Hyperparameter search
- [ ] Optuna integration with Random sampler, TPE sampler (Bayesian optimisation), and pruners for early stopping of bad trials
- [ ] Walk-forward validation / time series cross-validation to replace single train/val split
- [ ] Window size as a swept hyperparameter (each window size generates different data shape and its own MLflow run)

### Training pipeline
- [ ] Retrain best config on train+val combined (90%) after config selection
- [ ] Train final production model on all available historical data
- [ ] Returns-based modelling experiment to reduce distribution shift sensitivity

### MLflow & model management
- [ ] MLflow Model Registry — register, version, and promote best models
- [ ] Prediction service using registered model URI instead of local file path
- [ ] Mock-based tests for `Predictor` (no real artifacts required on fresh clone)

### Deployment & monitoring
- [ ] Cloud deployment (AWS/GCP) with Docker registry push
- [ ] Prometheus + Grafana monitoring for prediction latency and data drift
