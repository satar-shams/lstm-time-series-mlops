# LSTM Time Series Forecasting – End-to-End MLOps Project

This project covers the early phase of an ML lifecycle: training an LSTM
forecasting model, containerizing it, and serving it through a REST API.

**Status:** Phase 0 / diagnostic phase, complete. This repo currently
represents a working but not yet production-structured pipeline. A
restructured version with proper experiment tracking, modular code, and
tests is planned as a separate effort.

## What's actually in this repo right now

- `train.py` — trains an LSTM on AAPL daily close prices (yfinance), tries
  three hyperparameter configs, saves the best one by test RMSE
- `app/main.py` — FastAPI service with a `/predict` endpoint that loads the
  saved model and returns a prediction
- `Dockerfile` — containerizes the API for serving
- `LSTM_MLflow/` — exploratory notebooks from an earlier MLflow-tracked
  training pass (not currently wired into `train.py`)
- `requirements.lock` — full pinned environment, generated via `pip freeze`,
  used to build the Docker image
- `requirements.txt` — top-level dependencies only

## Model

- Architecture: 2-layer LSTM (64 units each) → Dense(128, relu) → Dropout(0.5)
  → Dense(1)
- Input: 30-day rolling window of closing price
- Train/test split: 90% / 10%, chronological (no shuffling)
- Best config RMSE: 3.19% (see note on limitations below)

## Tech stack

- TensorFlow 2.18.0 / Keras 3.15.0
- FastAPI + Uvicorn
- Docker
- scikit-learn (StandardScaler, metrics)
- yfinance (data source)

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
python train.py          # trains and saves models/best_model.keras
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## How to run with Docker

```bash
docker build -t lstm-api .
docker run -p 8000:8000 lstm-api
```

Test it:

```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]}'
```

Note: `/predict` currently returns the model's raw scaled output, not an
inverse-transformed price. This is a known gap — the fitted `StandardScaler`
is not yet saved or loaded alongside the model. See Known Limitations.

## Incident: Keras/TensorFlow version drift breaking Docker serving

### Symptom

Model trained and saved successfully (`best_model.keras`), but loading it
inside the Docker container at serve time failed with:

```
TypeError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

### Root cause

`tensorflow==2.18.0` does not pin a specific Keras version. Since TensorFlow
2.16, Keras 3 ships and updates as an independently versioned package. The
model was trained in one environment that resolved `keras==3.15.0`, while
the Docker image — built from a `requirements.txt` that only pinned
`tensorflow` — resolved a different Keras 3.x release at build time. A newer
`Dense` layer config key (`quantization_config`) wasn't recognized by the
older version's `Dense.__init__()`, breaking model deserialization on load.

This was not a Docker or CUDA issue. The CUDA driver warnings visible in the
logs are expected CPU-fallback behavior in a container with no GPU
passthrough, and were a red herring during initial diagnosis.

### Fix

- Pinned `keras==3.15.0` explicitly alongside `tensorflow==2.18.0`
- Generated `requirements.lock` via `pip freeze` to capture the fully
  resolved environment, not just top-level packages
- Rebuilt the Docker image installing from `requirements.lock` rather than
  the loose `requirements.txt`, so the container environment matches the
  training environment exactly
- Matched the Dockerfile's Python base image (3.12) to the training venv's
  Python version, since wheel resolution is also Python-version-dependent

### Lesson

Pinning a top-level framework (`tensorflow==X`) does not pin its
sub-dependencies once that framework decouples its own versioning, as Keras
did starting at TensorFlow 2.16. Train/serve environment parity has to be
enforced at the full dependency-tree level (a lockfile), not just on the
headline package.

## Known limitations (not yet fixed)

- **Unscaled predictions**: `/predict` returns the raw model output in
  `StandardScaler`-transformed space, not real price units. The fitted
  scaler needs to be persisted (e.g. via `pickle` or `joblib`) and loaded
  in `app/main.py` to inverse-transform the output before returning it.
- **Model selection uses the test set**: the three hyperparameter configs
  in `train.py` are compared directly on `X_test`/`y_test`, and the
  best-performing one is selected and saved. This means the test set is
  doing double duty as both a model-selection signal and a final
  performance estimate, which optimistically biases the reported RMSE.
  A proper train/validation/test split is needed so the test set is only
  touched once, after a config has already been chosen on validation data.
- **No experiment tracking wired into `train.py`**: `LSTM_MLflow/` contains
  an earlier exploratory pass with MLflow tracking, but the current
  `train.py` does not log params, metrics, or artifacts anywhere.
- **No automated tests** for data loading, the training pipeline, or the
  API endpoints.
- **Notebook/script duplication**: training logic currently exists in both
  `train.py` and `LSTM_MLflow/LSTM_Training_MLflow.ipynb`, written
  independently of each other.

These are the intended scope of a follow-up restructuring pass, not
oversights to be fixed quietly — listed here deliberately so the current
state of the project is accurate.