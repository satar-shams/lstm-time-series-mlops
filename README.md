# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, Optuna hyperparameter search, MLflow experiment tracking, three-way
train/val/test split, early stopping, adaptive learning rate scheduling,
gradient clipping, containerised serving, structured logging, input
validation, and automated tests.

> **Status:** Phase 2 active on `dev`. Optuna search and MLflow tracking
> are fully wired in. Walk-forward validation, MLflow Model Registry, and
> cloud deployment are next.

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
│   │   └── trainer.py             # TimeSeriesTraining — Optuna search, callbacks, MLflow logging
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
| Architecture | Input(30,1) → LSTM(64) → LSTM(64) → Dense(128, relu) → Dropout(0.3) → Dense(1) |
| Input | 30-day rolling window of adjusted closing price |
| Target | Next-day closing price |
| Split | 70% train / 15% validation / 15% test, chronological — no shuffling |
| Scaler | `StandardScaler` fit on training data only (no leakage into val or test) |
| Optimizer | Adam with gradient clipping (`clipnorm`) |
| Callbacks | `EarlyStopping` (patience=10, restore\_best\_weights) + `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Hyperparameter search | Optuna TPE sampler — each trial logged as a named MLflow run |
| Model selection criterion | Validation RMSE% — test set never touched during search |
| Final model training | Best config retrained on train+val combined (85%), evaluated on test |
| Experiment tracking | MLflow with SQLite backend and dedicated artifact store |

---

## Results

| Stage | Period | RMSE% | Notes |
|---|---|---|---|
| Best trial (validation) | 2021–2024 | 2.84% | Trial 8 of 10 |
| Final model (test) | 2024–2026 | 4.56% | Retrained on train+val, evaluated once |

**Best hyperparameters found (10-trial exploratory run):**

| Parameter | Value |
|---|---|
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `lstm_units` | 64 |
| `dense_units` | 128 |
| `dropout_rate` | 0.3 |
| `clip_norm` | 0.5 |
| `epochs` (early stopped at) | 36 |

The test RMSE% (4.56%) is substantially better than the baseline grid search
result (14–17%) due to two factors: Optuna finding genuinely better
hyperparameters (notably `dropout_rate=0.3` and `clip_norm=0.5` vs. defaults),
and the final model being retrained on train+val combined (85% of data) rather
than train only (70%). The remaining gap between validation and test is
attributable to **distribution shift** — the 2024–2026 test period represents
a structurally different price regime from the 2010–2021 training period.

These results reflect a **10-trial exploratory run**. A full production sweep
with more trials and walk-forward validation is scoped for the next phase.

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow 2.18.0 / Keras 3.15.0 |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn `StandardScaler` |
| Hyperparameter search | Optuna 4.9.0 (TPE sampler) |
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

Fetches AAPL data, runs an Optuna hyperparameter search where each trial is
logged as a named MLflow run with params, val metrics, model artifact, and
scaler artifact. After the study completes, the best config is loaded from
MLflow, a final model is retrained on train+val combined, evaluated on the
held-out test set, and logged as `final_model` in MLflow.

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
defaults. `HYPER_PARAMS` defines the Optuna search space — comment/uncomment
entries to control which parameters are tuned vs. held at their default.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.85
OPTUNA_TRIALS = 20

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
    "learning_rate": [0.01, 0.001, 0.0001],
    "lstm_units": [32, 64, 128],
    "dense_units": [64, 128, 256],
    "dropout_rate": [0.3, 0.5, 0.7],
    "clip_norm": [0.5, 1.0, 2.0, 5.0],
}
```

> Parameters removed from `HYPER_PARAMS` automatically fall back to their
> value in `DEFAULTS_PARAMS`. Every parameter in `HYPER_PARAMS` must have
> a corresponding entry in `DEFAULTS_PARAMS`.

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
| **Exploratory search only** | 10-trial Optuna run with categorical search space. Production-grade search requires more trials, `suggest_float`/`suggest_int` for continuous params, and Optuna pruners. |
| **Distribution shift on test set** | Model trained on 2010–2021 data performs worse on 2024–2026 data due to price regime change. Walk-forward validation and returns-based modelling are planned mitigations. |
| **MLflow Model Registry not yet used** | Model artifacts are logged per run but not registered or versioned in the MLflow Model Registry. |
| **Predictor test requires trained artifacts** | `tests/test_predictor.py` loads real model and scaler files from `models/`, which are gitignored. Proper fix is mocking `load_model`/`joblib.load`; deferred for now. |
| **No multi-step forecasting** | The model predicts one day ahead. Multi-day forecasting via recursive window-sliding is a planned `Predictor` extension. |

---

## Phase 2 roadmap

### Hyperparameter search
- [ ] Optuna `suggest_float` / `suggest_int` for continuous search spaces
- [ ] Optuna pruners for early stopping of unpromising trials
- [ ] Walk-forward validation / time series cross-validation
- [ ] Window size as a swept hyperparameter

### Training pipeline
- [ ] Train final production model on all available historical data
- [ ] Returns-based modelling experiment to reduce distribution shift sensitivity

### MLflow & model management
- [ ] MLflow Model Registry — register, version, and promote best models
- [ ] Prediction service using registered model URI instead of local file path
- [ ] Mock-based tests for `Predictor` (no real artifacts required on fresh clone)

### Deployment & monitoring
- [ ] Cloud deployment (AWS/GCP) with Docker registry push
- [ ] Prometheus + Grafana monitoring for prediction latency and data drift
