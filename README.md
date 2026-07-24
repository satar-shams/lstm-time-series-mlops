# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, Optuna hyperparameter search, MLflow experiment tracking and Model
Registry, three-way train/val/test split, early stopping, adaptive learning
rate scheduling, gradient clipping, reproducible training via fixed seeding,
containerised serving, structured logging, input validation, and automated
tests.

> **Status:** Phase 2 milestone (`v1.3.0`). Optuna search, MLflow tracking,
> and alias-based Model Registry are fully wired in. Walk-forward validation
> and cloud deployment are next.

---

## Project structure

lstm-time-series-mlops/
├── src/
│ ├── config.py # Single source of truth for all constants and defaults
│ ├── data/
│ │ ├── loader.py # StockLoader — yfinance fetch + validation
│ │ └── preprocessor.py # TimeSeriesPreprocessor — scaling, windowing, three-way split
│ ├── models/
│ │ ├── lstm_model.py # LSTMForecaster — architecture + compilation
│ │ └── base_model.py # Reserved for a shared model interface if more model types are added
│ ├── training/
│ │ ├── trainer.py # TimeSeriesTraining — orchestrates the full pipeline
│ │ ├── optuna_tuner.py # LSTMOptuna — hyperparameter search, per-trial MLflow logging
│ │ ├── single_model_trainer.py # SingleModelTrainer — trains one model for a given config
│ │ ├── evaluator.py # LSTMEvaluator — real-price MAE/RMSE/RMSE%
│ │ ├── mlflow_manager.py # MLFlowManager — param/metric/model/artifact logging
│ │ ├── model_registry.py # ModelRegistry — registration, alias-based versioning, model loading
│ │ ├── callbacks.py # EarlyStopping + ReduceLROnPlateau factory
│ │ ├── summary.py # TrainingSummary — structured console output
│ │ └── utils.py # set_random_seed() — reproducibility across runs
│ └── inference/
│ └── predictor.py # Predictor — loads model/scaler from MLflow Registry, predicts
├── app/
│ ├── main.py # FastAPI application entrypoint
│ ├── example.py # Sample prediction payload for Swagger UI
│ ├── api/
│ │ └── routes/
│ │ ├── health.py # GET /api/v1/health
│ │ └── prediction.py # POST /api/v1/predict
│ ├── core/
│ │ ├── config.py # APISettings — env-based host/port/version via pydantic-settings
│ │ ├── exceptions.py # ModelLoadError, PredictionFailedError
│ │ ├── exception_handlers.py # Maps custom exceptions to clean JSON error responses
│ │ └── logger.py # Structured JSON logger
│ └── schemas/
│ ├── health.py # HealthResponse
│ └── prediction.py # PredictRequest (length-validated), PredictResponse
├── tests/
│ ├── test_loader.py
│ └── test_lstm_model.py
├── scripts/
│ └── train.py
├── notebooks/
│ ├── exploration.ipynb
│ ├── LSTM_Training_MLflow.ipynb
│ ├── Load_Save_registered_Model.ipynb
│ └── train_legacy.py # Original flat training script, kept for reference
├── models/ # Gitignored — populated by training runs
├── mlruns/ # Gitignored — MLflow local run metadata
├── mlartifacts/ # Gitignored — MLflow model and scaler artifacts
├── mlflow.db # Gitignored — MLflow SQLite backend
├── .env # Gitignored — local environment variables
├── .env.example # Committed template for required environment variables
├── Dockerfile
├── requirements.txt # Top-level production dependencies
├── requirements.lock # Full pinned environment (used by Docker)
├── requirements-dev.txt # Dev-only dependencies (pytest)
└── README.md


---

## Model

| Property | Value |
|---|---|
| Architecture | Input(30,1) → LSTM → LSTM → Dense(relu) → Dropout → Dense(1) — units tuned by Optuna |
| Input | 30-day rolling window of adjusted closing price |
| Target | Next-day closing price |
| Split | 90% train / 5% validation / 5% test, chronological — no shuffling |
| Scaler | `StandardScaler` fit on training data only (no leakage into val or test) |
| Optimizer | Adam with gradient clipping (`clipnorm`, tuned) |
| Callbacks | `EarlyStopping` (patience=10, restore\_best\_weights) + `ReduceLROnPlateau` (factor=0.5, patience=5) |
| Hyperparameter search | Optuna TPE sampler, 25 trials — each logged as a named MLflow run |
| Model selection criterion | Validation RMSE% — **test set is never used for search or selection** |
| Training pipeline | Three tiers: search (train, eval on val) → candidate (train+val, eval on test) → production (all data) |
| Reproducibility | Fixed seed + `tf.config.experimental.enable_op_determinism()` |
| Model management | MLflow Model Registry, alias-based versioning (no deprecated stage API) |

---

## Results

**Full 25-trial search:**

| Metric | Value | Meaning |
|---|---|---|
| Optimization RMSE% (validation) | 1.94% | Used for trial selection — trial 22 |
| Trial's own test RMSE% | 1.52% | Diagnostic only, not used for selection or reported as the final result |
| Final test RMSE% (candidate model) | 3.11% | Retrained on train+val, evaluated once — this is the honest result |
| **Generalization gap** | **+1.17%** | Final test RMSE% − optimization RMSE% |

**Best hyperparameters found:**

| Parameter | Value |
|---|---|
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `lstm_units` | 128 |
| `dense_units` | 256 |
| `dropout_rate` | 0.3 |
| `clip_norm` | 5.0 |
| `epochs` (early stopped at) | 30 |

The production model is subsequently retrained on **all available data**
(train + val + test) using this configuration, since no further held-out
evaluation is needed once the honest test score above has been reported.

### A note on the two test RMSE% numbers

Two different test-set numbers appear in training output, and only one of
them should be trusted as the reported result. Each Optuna trial's own
model — trained with early stopping/LR scheduling active, on train data
only — is evaluated on the test set purely for diagnostic visibility, and
that score is logged per trial. It is **not** used to select the winning
trial (selection uses validation RMSE% only) and it is noisy: across
different trials in the same search, this per-trial test score sometimes
lands above validation RMSE% and sometimes below it, which is expected
variance from a single train/early-stop run rather than a meaningful
signal on its own.

The number reported as "Final Test RMSE%" is different and is the one
that matters: it comes from the **candidate model**, retrained from
scratch on train+val combined, using a fixed epoch count (the winning
trial's `best_epoch`, not early stopping), then evaluated on the test set
exactly once. This is the only test-set number used anywhere in this
pipeline for reporting or decision-making, and it is what the
generalization gap above is computed from.

### Investigation: closing the validation-test gap

An earlier version of this pipeline used a 70/15/15 split and saw
**generalization gaps as large as +19.86%** between validation and test
RMSE% — a config that looked excellent during search performed far worse
on held-out data. Two hypotheses were tested in order, not assumed:

1. **Callback state loss during retraining.** The candidate model is
   retrained from scratch after search, using only the winning
   hyperparameters — but Optuna's `EarlyStopping`/`ReduceLROnPlateau`
   also shape the training process (best epoch, LR schedule). To test
   whether this mattered, the original per-trial model (with callbacks)
   was compared directly against the retrained model (fixed epoch count,
   no callbacks) on the same test set. The gap between them was small
   (7.48% vs. 6.26%), ruling this out as the primary cause.
2. **Insufficient training data.** With only 70% of ~4,000 trading days
   available for training, the model had comparatively little history to
   learn from. Changing the split to 90% train / 5% val / 5% test gave the
   model substantially more data while still preserving genuinely held-out
   evaluation. This produced a dramatic improvement: the generalization
   gap fell from +19.86% to +0.89% in initial testing, and +1.17% in the
   full 25-trial run reported above.

This investigation is a deliberate example of testing one variable at a
time before changing another — worth noting since a smaller test set (5%
of history, roughly 200 trading days) also means a noisier, less certain
estimate of generalization than the earlier 15% test set. The improvement
is real but should be read in that context, not as a fully closed question.

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow 2.18.0 / Keras 3.15.0 |
| Data | yfinance, pandas, NumPy |
| Preprocessing | scikit-learn `StandardScaler` |
| Hyperparameter search | Optuna 4.9.0 (TPE sampler) |
| Experiment tracking | MLflow 3.14.0 (SQLite backend, Model Registry) |
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

Run this in a **separate terminal** before training.

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:./mlartifacts
```

The MLflow UI (including the Model Registry) is available at
`http://127.0.0.1:5000`.

### 3. Train the model

```bash
python -m src.training.trainer
```

This runs the full pipeline: fetches AAPL data, runs an Optuna search
(each trial logged as an MLflow run with params, validation and test
metrics, model and scaler artifacts), retrains the best config on
train+val as a candidate model, evaluates it honestly on the held-out
test set, registers it in the MLflow Model Registry under the alias
`production`, then retrains a final production model on all available
data and saves it locally.

### 4. Configure the API environment

The API reads its host and port from environment variables via
`pydantic-settings`. Copy the example file and adjust if needed:

```bash
cp .env.example .env
```

`.env.example` contents:

APP_HOST=0.0.0.0
APP_PORT=8000

`.env` is gitignored — each environment (local, CI, production) supplies
its own values. `.env.example` documents the required keys with safe
placeholder values.

### 5. Run the API locally

> **MLflow must be running before starting the API.** `Predictor` loads
> the production model and scaler directly from the MLflow Model Registry
> at startup — if the MLflow server (step 2) isn't running, the API will
> fail to start.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. Test the endpoints

```bash
curl http://localhost:8000/
curl http://localhost:8000/api/v1/health
```

```bash
curl -X POST http://localhost:8000/api/v1/predict \
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

Swagger UI (`/docs`) includes a pre-filled example payload for quick testing.

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
defaults. `HYPER_PARAMS` defines the Optuna search space.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.95
OPTUNA_TRIALS = 25
SEED = 42

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction"
MLFLOW_MODEL_NAME = "LSTMStockPredictor"

### Model Registry: candidate vs. production

Every training run registers two separate model versions under
`LSTMStockPredictor`, linked by identical hyperparameters but trained on
different data:

- **`@candidate`** — trained on train+validation, evaluated once on the
  held-out test set. This version carries the only honest, trustworthy
  performance number in the pipeline (see "Final Test RMSE%" in the
  training summary). It is not served in production.
- **`@production`** — trained on train+validation+test (all available
  history), using the same configuration the candidate proved sound. This
  version has no held-out test metrics of its own — there is no unseen
  data left to evaluate it against — but it is the most data-informed
  model available, and it is the one `Predictor` loads and serves.

In short: `@candidate` tells you how good the configuration is;
`@production` is what actually answers requests, trained on everything
that configuration has proven itself against.

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

> Every parameter in `HYPER_PARAMS` must have a corresponding entry in
> `DEFAULTS_PARAMS`. Removing a parameter from `HYPER_PARAMS` (but keeping
> its default) stops it from being tuned without breaking the pipeline.

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
| **Single train/val/test split** | The 90/5/5 split uses one fixed historical window. Walk-forward validation (rolling multiple train/test windows through time) would give a more robust generalization estimate. |
| **Small test set** | 5% of ~4,000 trading days (~200 days) gives a noisier held-out estimate than the earlier 15% split. The current +1.17% generalization gap should be read with this in mind. |
| **Categorical search space only** | Optuna currently uses `suggest_categorical` over discrete lists. `suggest_float`/`suggest_int` with continuous ranges and pruners for early trial termination are not yet implemented. |
| **Test metrics logged (not used) during search** | Every Optuna trial logs test-set metrics for diagnostic visibility, but trial selection strictly uses validation RMSE% only. Test metrics are never fed into the objective function. |
| **Predictor test requires trained artifacts** | `tests/test_predictor.py` loads real model and scaler files from `models/`, which are gitignored. Proper fix is mocking `load_model`/`joblib.load`; deferred for now. |
| **No multi-step forecasting** | The model predicts one day ahead. Multi-day forecasting via recursive window-sliding is a planned `Predictor` extension. |
| **Placeholder test/model files empty** | `tests/test_loader.py`, `tests/test_lstm_model.py`, `src/models/base_model.py` are reserved for future work but currently contain no code. |
| **API startup hangs rather than failing fast without MLflow** | If the MLflow server is unreachable, `uvicorn` startup does not exit cleanly (may require `pkill`/force-kill). Root cause is likely MLflow client's internal retry/backoff on connection failure. A startup connection-timeout check is a reasonable future improvement, not yet implemented. |

---

## Phase 2 roadmap

### Hyperparameter search
- [ ] Optuna `suggest_float` / `suggest_int` for continuous search spaces
- [ ] Optuna pruners for early stopping of unpromising trials
- [ ] Walk-forward validation / time series cross-validation
- [ ] Window size as a swept hyperparameter

### Model management
- [ ] Load model for inference by MLflow Model Registry alias, not local file path
- [ ] Mock-based tests for `Predictor` (no real artifacts required on fresh clone)

### Deployment & monitoring
- [ ] Cloud deployment (AWS/GCP) with Docker registry push
- [ ] Prometheus + Grafana monitoring for prediction latency and data drift
- [ ] Returns-based modelling experiment as an alternative to price-level prediction
