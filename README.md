# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, config-driven hyperparameter sweeps, containerised serving, structured
logging, input validation, and automated tests.

> **Status:** Phase 1 complete. The pipeline covers the full lifecycle from
> raw data to a served, containerised API. MLflow experiment tracking and a
> proper train/validation/test split are scoped for Phase 2.

---

## Project structure

```
lstm-time-series-mlops/
├── src/
│   ├── config.py                  # Single source of truth for all constants
│   ├── data/
│   │   ├── loader.py              # StockLoader — yfinance fetch + validation
│   │   └── preprocessor.py        # TimeSeriesPreprocessor — scaling, windowing
│   ├── models/
│   │   └── lstm_model.py          # LSTMForecaster — architecture only
│   ├── training/
│   │   └── trainer.py             # TimeSeriesTraining — grid search + evaluation
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
| Train / test split | 90% / 10%, chronological — no shuffling |
| Scaler | `StandardScaler` fit on training data only (no leakage) |
| Hyperparameter search | Grid search via `itertools.product` over `config.HYPER_PARAMS` |
| Best observed RMSE% | ~2% (varies per run — see Known Limitations) |

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow 2.18.0 / Keras 3.15.0 |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn `StandardScaler` |
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

### 2. Train the model

```bash
python -m src.training.trainer
```

This fetches AAPL data via yfinance, runs a hyperparameter grid search, saves
the best model to `models/best_model.keras` and the fitted scaler to
`models/scaler.bin`.

### 3. Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Test the endpoints

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
- `create_windows` boundary cases (first, middle, and last window) with
  hand-verified expected arrays
- Scaler fit/transform/inverse_transform round-trip correctness
- `Predictor` input length validation (`ValueError` on wrong-length input)

---

## Configuration

All constants live in `src/config.py`. To change the ticker, date range,
default architecture parameters, or active hyperparameter sweep values,
edit that file — nothing is hardcoded elsewhere.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
SPLIT_SIZE = 0.9

HYPER_PARAMS = {
    "epochs": [20, 30, 40],
    "batch_size": [32, 64],
    # "learning_rate": [0.01, 0.001, 0.0001],
    # "lstm_units": [32, 64, 128],
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
| **Model selection uses the test set** | Hyperparameter configs are compared directly on `X_test`/`y_test`. A proper train/validation/test three-way split is needed to avoid optimistic RMSE bias. Scoped for Phase 2. |
| **No MLflow experiment tracking** | `notebooks/` contains an earlier exploratory MLflow pass. Wiring it into `src/training/trainer.py` (logging params, metrics, and model artifacts per run) is the primary Phase 2 goal. |
| **Predictor test requires trained artifacts** | `tests/test_predictor.py` loads real model and scaler files from `models/`, which are gitignored. A fresh clone without a prior training run will fail this test. Proper fix is mocking `load_model`/`joblib.load`; deferred for now. |
| **No multi-step forecasting** | The model predicts one day ahead. Multi-day forecasting via recursive window-sliding is a planned `Predictor` extension. |

---

## Phase 2 roadmap

- [ ] MLflow experiment tracking wired into `src/training/trainer.py`
- [ ] Train / validation / test three-way split
- [ ] Cloud deployment (AWS/GCP) with Docker registry push
- [ ] Prometheus + Grafana monitoring for prediction latency and drift
- [ ] Mock-based tests for `Predictor` (no real artifacts required)
