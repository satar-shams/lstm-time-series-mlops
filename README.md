# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, Optuna hyperparameter search, MLflow experiment tracking and Model
Registry, three-way train/val/test split, early stopping, adaptive learning
rate scheduling, gradient clipping, reproducible training via fixed seeding,
a fully containerised pipeline with separate images per service, structured
logging, input validation, automated tests, and continuous integration.

> **Status:** Phase 2 milestone. Optuna search, MLflow tracking, alias-based
> Model Registry, a fully Dockerised training/serving pipeline (Docker
> Compose, one image per service), and a GitHub Actions CI pipeline are
> complete. Walk-forward validation, continuous hyperparameter ranges, and
> cloud deployment are next.

---

## Project structure

```
lstm-time-series-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: unit tests + build all three images
├── src/
│   ├── config.py                   # Single source of truth for all constants and defaults
│   ├── data/
│   │   ├── loader.py               # StockLoader — yfinance fetch + validation
│   │   └── preprocessor.py         # TimeSeriesPreprocessor — scaling, windowing, three-way split
│   ├── models/
│   │   ├── lstm_model.py           # LSTMForecaster — architecture + compilation
│   │   └── base_model.py           # Empty — reserved for a shared model interface if more model types are added
│   ├── training/
│   │   ├── trainer.py              # TimeSeriesTraining — orchestrates the full pipeline
│   │   ├── optuna_tuner.py         # LSTMOptuna — hyperparameter search, per-trial MLflow logging
│   │   ├── single_model_trainer.py # SingleModelTrainer — trains one model for a given config
│   │   ├── evaluator.py            # LSTMEvaluator — real-price MAE/RMSE/RMSE%
│   │   ├── mlflow_manager.py       # MLFlowManager — param/metric/model/artifact logging
│   │   ├── model_registry.py       # ModelRegistry — registration, alias-based versioning, model loading
│   │   ├── callbacks.py            # EarlyStopping + ReduceLROnPlateau factory
│   │   ├── summary.py              # TrainingSummary — structured console output
│   │   └── utils.py                # set_random_seed() — reproducibility across runs
│   └── inference/
│       └── predictor.py            # Predictor — loads model/scaler from MLflow Registry, predicts
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── example.py                  # Sample prediction payload for Swagger UI
│   ├── api/
│   │   └── routes/
│   │       ├── health.py           # GET /api/v1/health
│   │       └── prediction.py       # POST /api/v1/predict
│   ├── core/
│   │   ├── config.py               # APISettings — env-based host/port/version via pydantic-settings
│   │   ├── exceptions.py           # ModelLoadError, PredictionFailedError
│   │   ├── exception_handlers.py   # Maps custom exceptions to clean JSON error responses
│   │   └── logger.py               # Structured JSON logger
│   └── schemas/
│       ├── health.py               # HealthResponse
│       └── prediction.py           # PredictRequest (length-validated), PredictResponse
├── tests/
│   ├── test_api.py                 # Health + predict routes (integration, requires MLflow)
│   ├── test_data_loader.py         # StockLoader (mocked yfinance)
│   ├── test_lstm_model.py          # Empty — reserved for future LSTMForecaster tests
│   ├── test_model_registry.py      # ModelRegistry (mocked MLflow)
│   ├── test_predictor.py           # Predictor unit tests (mocked MLflow)
│   ├── test_predictor_integration.py # Predictor live end-to-end (integration, requires MLflow)
│   ├── test_preprocessor.py        # Windowing, three-way split, scaler round-trip
│   └── test_loader.py              # Empty — reserved for future StockLoader tests
├── scripts/
│   └── train.py
├── notebooks/
│   ├── exploration.ipynb
│   ├── LSTM_Training_MLflow.ipynb
│   ├── Load_Save_registered_Model.ipynb
│   └── train_legacy.py             # Original flat training script, kept for reference
├── models/                         # Gitignored — populated by non-Docker training runs
├── mlruns/                         # Gitignored — legacy local MLflow run metadata
├── mlartifacts/                    # Gitignored — MLflow model and scaler artifacts (Docker volume)
├── mlflow.db                       # Gitignored — MLflow SQLite backend (Docker volume)
├── .env                            # Gitignored — local environment variables (non-Docker path)
├── .env.example                    # Committed template for required environment variables
├── .dockerignore
├── Dockerfile.api                  # API service image (fastapi, tensorflow, mlflow client)
├── Dockerfile.trainer               # Trainer service image (+ optuna, yfinance, pandas)
├── Dockerfile.mlflow                # MLflow tracking server image
├── docker-compose.yml               # Orchestrates mlflow, trainer, api
├── requirements.api.lock            # Pinned dependencies for the API image
├── requirements.trainer.lock        # Pinned dependencies for the trainer image
├── requirements.mlflow.lock         # Pinned dependencies for the MLflow image
├── requirements-dev.txt             # Dev-only dependencies (pytest, httpx)
└── README.md
```

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
| Model management | MLflow Model Registry, alias-based versioning (`candidate` / `production`) |

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
| Experiment tracking | MLflow 3.14.0 (SQLite backend, `--serve-artifacts` proxy, Model Registry) |
| Serving | FastAPI + Uvicorn |
| Persistence | joblib (scaler), Keras native `.keras` format (model) |
| Containerisation | Docker + Docker Compose — separate images per service (`api`, `trainer`, `mlflow`) |
| CI | GitHub Actions — unit tests + build all three images on every push |
| Tests | pytest, httpx |

---

## Quickstart (Docker)

The entire pipeline — MLflow, training, and the API — runs through
Docker Compose, with each service built from its own Dockerfile and its
own pinned dependency set. No local Python environment is required for
this path.

### 1. Clone the repository

```bash
git clone https://github.com/satar-shams/lstm-time-series-mlops.git
cd lstm-time-series-mlops
```

### 2. Build the images

```bash
docker compose build
```

This builds three separate images:
- **`api`** — FastAPI + TensorFlow + MLflow client (inference only, no
  Optuna, no yfinance, no pandas)
- **`trainer`** — TensorFlow + MLflow + Optuna + yfinance + pandas
- **`mlflow`** — MLflow only, the smallest of the three

Splitting dependencies this way keeps the `api` image free of
training-only packages it never uses at runtime.

### 3. Start the MLflow server

```bash
docker compose up -d mlflow
```

Wait until the service reports healthy:

```bash
docker compose ps
```

Expected:

```
mlflow   Up (healthy)
```

The MLflow UI is available at `http://localhost:5000`.

> **Note:** Training logs a "View run..." link pointing at
> `http://mlflow:5000` (the internal Docker service name), which is not
> reachable from a browser on your host. Use `http://localhost:5000`
> instead to browse the MLflow UI — see Known Limitations.

### 4. Train the model

```bash
docker compose run --rm trainer
```

This runs the full pipeline inside the `trainer` container: fetches AAPL
data, runs a 25-trial Optuna search, trains and registers a candidate
model (tested on the held-out test set), then a production model
(trained on all available data). Both are logged to the MLflow Model
Registry under the `LSTMStockPredictor` name with the
`candidate`/`production` aliases.

Monitor progress in the MLflow UI at `http://localhost:5000`.

### 5. Start the prediction API

```bash
docker compose up -d api
```

The API loads the production model directly from the MLflow Model
Registry at startup — no local model files are used.

Verify it started successfully:

```bash
docker compose logs api
```

Expected:

```
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

### 6. Test the endpoints

Swagger UI: `http://localhost:8000/docs`

```bash
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

Input must contain exactly 30 float values. Sending the wrong number
returns a clear `422` error:

```json
{"detail": [{"type": "too_short", "loc": ["body", "data"], "msg": "List should have at least 30 items after validation, not 29"}]}
```

### Daily development (no retraining)

Once a production model is registered, you don't need to retrain to
restart the stack:

```bash
docker compose up -d mlflow api
```

The API automatically loads the latest registered production model on
startup. Retrain only when you want to create a new model version:

```bash
docker compose run --rm trainer
```

### Stop everything

```bash
docker compose down
```

---

## Docker architecture

```
                 Docker Compose
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
   trainer            mlflow             api
  (one-shot)       (long-running)   (long-running)
   Dockerfile.      Dockerfile.       Dockerfile.
   trainer           mlflow             api
        │               │                │
        └──────► Model Registry ◄────────┘
                        │
                        ▼
              MLflow Artifact Store
                (mlartifacts/ volume)
```

Each service is built from its own Dockerfile and its own pinned
requirements file, rather than one shared image:

- **`mlflow`** (`Dockerfile.mlflow`, `requirements.mlflow.lock`) — the
  central tracking server, model registry, and artifact store (SQLite
  backend, `--serve-artifacts` proxy mode). The smallest image, since it
  only needs `mlflow` itself.
- **`trainer`** (`Dockerfile.trainer`, `requirements.trainer.lock`) —
  runs once (`docker compose run --rm trainer`), trains and registers
  models, then exits. Includes Optuna, yfinance, and pandas, which the
  API never needs.
- **`api`** (`Dockerfile.api`, `requirements.api.lock`) — long-running
  FastAPI service, loads the production model from the registry at
  startup, serves predictions. Excludes all training-only dependencies.

This split was made deliberately after measuring that a single shared
image was carrying dependencies neither service actually used at
runtime. `docker-compose.yml` handles only orchestration (networking,
environment variables, dependencies) — each `Dockerfile`'s `CMD` remains
the single source of truth for how its service starts, keeping local
Docker, CI, and any future cloud deployment consistent.

---

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main`, `dev`, and
`ci-cd-test`, and on pull requests targeting `main`/`dev`:

1. Installs dependencies from `requirements.trainer.lock` +
   `requirements.api.lock` + `requirements-dev.txt` (the union covers
   everything the test suite imports, without a separate, manually
   maintained requirements file to keep in sync)
2. Runs the unit test suite (`--ignore=tests/test_api.py
   --ignore=tests/test_predictor_integration.py` — the two tests that
   require a live MLflow server are excluded, see Run Tests)
3. Builds all three Docker images (`api`, `trainer`, `mlflow`) to confirm
   each `Dockerfile` still builds cleanly

This is CI only — image builds are not yet pushed to a registry or
deployed anywhere. That's the next phase (see roadmap).

---

## Alternative: run without Docker

For quick local debugging without containers.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.trainer.lock -r requirements.api.lock -r requirements.mlflow.lock
```

Start MLflow directly:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:./mlartifacts
```

Copy and configure environment variables:

```bash
cp .env.example .env
```

Train:

```bash
python -m src.training.trainer
```

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **MLflow must be running before starting the API.** `Predictor` loads
> the production model and scaler directly from the MLflow Model Registry
> at startup — if MLflow isn't reachable, the API will fail to start
> (though see Known Limitations regarding a startup hang rather than a
> clean failure).

---

## Run tests

```bash
pip install -r requirements.trainer.lock -r requirements.api.lock
pip install -r requirements-dev.txt
```

**Unit tests only** (fast, no external dependencies — what CI runs):

```bash
python -m pytest tests/ -v --ignore=tests/test_api.py --ignore=tests/test_predictor_integration.py
```

**Full suite, including integration tests** (requires the MLflow server
running with a registered `production` model):

```bash
python -m pytest tests/ -v
```

`test_api.py` and `test_predictor_integration.py` construct a real
`Predictor()` at import time, which connects to the MLflow Model
Registry. Both are marked `@pytest.mark.integration`, but since the
connection happens at import — before pytest's marker filtering runs —
they must be excluded via `--ignore`, not just deselected with
`-m "not integration"`. See Known Limitations.

Tests cover:
- `StockLoader.fetch()` — success, empty-response handling, MultiIndex
  column flattening (all mocked, no live yfinance calls)
- `TimeSeriesPreprocessor` — three-way split boundaries, windowing
  (first/middle/last), scaler fit/transform/inverse\_transform round-trip
- `LSTMForecaster` — architecture, layer types, prediction shape,
  compilation
- `ModelRegistry` — registration, alias assignment, model/scaler loading
  (all mocked, no live MLflow calls)
- `Predictor` — success path, model-load failure, prediction failure
  (mocked) and a live end-to-end prediction (integration, requires MLflow)
- FastAPI routes — health check, successful prediction, invalid input
  length (integration, requires MLflow)

---

## Configuration

All constants and default parameter values live in `src/config.py`.
`DEFAULTS_PARAMS` is the single source of truth for all model and training
defaults. `HYPER_PARAMS` defines the Optuna search space. `MLFLOW_TRACKING_URI`
and `MLFLOW_EXPERIMENT_NAME` read from environment variables, with local
defaults as fallback — Docker Compose sets these explicitly per service.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.95
OPTUNA_TRIALS = 25
SEED = 42

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "LSTM Stock Prediction Production")
MLFLOW_MODEL_NAME = "LSTMStockPredictor"

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

### Model Registry: candidate vs. production

Every training run registers two separate model versions under
`LSTMStockPredictor`, linked by identical hyperparameters but trained on
different data:

- **`@candidate`** — trained on train+validation, evaluated once on the
  held-out test set. This version carries the only honest, trustworthy
  performance number in the pipeline (see "Final Test RMSE%" above). It
  is not served in production.
- **`@production`** — trained on train+validation+test (all available
  history), using the same configuration the candidate proved sound. This
  version has no held-out test metrics of its own — there is no unseen
  data left to evaluate it against — but it is the most data-informed
  model available, and it is the one `Predictor` loads and serves.

In short: `@candidate` tells you how good the configuration is;
`@production` is what actually answers requests, trained on everything
that configuration has proven itself against.

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
requirements file that only pinned `tensorflow` — resolved a different
Keras 3.x release at build time. The newer `Dense` layer config key
(`quantization_config`) was not recognised by the older version's
`Dense.__init__()`, breaking model deserialisation.

The CUDA warnings visible in container logs are unrelated — expected
CPU-fallback behaviour with no GPU passthrough.

### Fix

- Pinned `keras==3.15.0` explicitly alongside `tensorflow==2.18.0`
- Generated a full lockfile via `pip freeze` to capture the complete
  resolved environment, not just top-level packages
- Rebuilt the Docker image from the lockfile (not a loose requirements
  file)
- Matched the Dockerfile's Python base image (`3.12-slim`) to the training
  venv's Python version

### Lesson

Pinning a top-level framework does not pin its sub-dependencies once that
framework decouples its own versioning. Train/serve environment parity must
be enforced at the full dependency-tree level via a lockfile.

---

## Incident record: stale MLflow artifact location after storage migration

### Symptom

After migrating the MLflow server to Docker (with `--serve-artifacts` and
`--artifacts-destination`), training runs appeared correctly in the MLflow
UI with params and metrics logged, but the Artifacts tab was empty — no
scaler, no model — for the pre-existing `LSTM Stock Prediction` experiment.

### Root cause

MLflow records an experiment's `artifact_location` permanently at the
moment the experiment is first created, and never updates it retroactively.
`LSTM Stock Prediction` was originally created during local (non-Docker)
training, so its stored `artifact_location` pointed at a path on the host
filesystem — valid then, but nonexistent inside the container's isolated
filesystem after the move to Docker. Every subsequent training run
correctly logged params/metrics (backend-store operations, unaffected)
but silently failed to write artifacts to a path that no longer existed.

### Fix

Created a new experiment (`LSTM Stock Prediction Production`) rather than
attempting to repair the old one's stored artifact location. A newly
created experiment picks up the *current* server's artifact configuration
at creation time, correctly resolving through the Docker-aware artifact
proxy. Verified independently first with a minimal `mlflow.log_artifact()`
test before retraining the full pipeline.

### Lesson

Any MLflow experiment created before a change to artifact storage
configuration retains its original, now-stale artifact location
indefinitely — this isn't Docker-specific, it applies to any artifact
storage migration (e.g. a later move to S3). The fix is always a new
experiment, never editing the old one's stored config. `MLFLOW_EXPERIMENT_NAME`
should be treated as versioned alongside significant infrastructure changes,
similar to how model versions are tracked.

---

## Known limitations

| Limitation | Notes |
|---|---|
| **Single train/val/test split** | The 90/5/5 split uses one fixed historical window. Walk-forward validation (rolling multiple train/test windows through time) would give a more robust generalization estimate. |
| **Small test set** | 5% of ~4,000 trading days (~200 days) gives a noisier held-out estimate than the earlier 15% split. The current +1.17% generalization gap should be read with this in mind. |
| **Categorical search space only** | Optuna currently uses `suggest_categorical` over discrete lists. `suggest_float`/`suggest_int` with continuous ranges and pruners for early trial termination are not yet implemented. |
| **Test metrics logged (not used) during search** | Every Optuna trial logs test-set metrics for diagnostic visibility, but trial selection strictly uses validation RMSE% only. Test metrics are never fed into the objective function. |
| **Predictor integration tests require --ignore, not just -m "not integration"** | `test_api.py` and `test_predictor_integration.py` construct `Predictor()` at import time, connecting to MLflow before pytest's marker filtering applies. Proper fix is lazy-loading `Predictor` inside a FastAPI dependency rather than at module level. |
| **No multi-step forecasting** | The model predicts one day ahead. Multi-day forecasting via recursive window-sliding is a planned `Predictor` extension. |
| **Placeholder test/model files empty** | `tests/test_loader.py`, `tests/test_lstm_model.py`, `src/models/base_model.py` are reserved for future work but currently contain no code. |
| **API startup hangs rather than failing fast without MLflow** | If the MLflow server is unreachable, `uvicorn` startup does not exit cleanly (may require force-kill). Root cause is likely MLflow client's internal retry/backoff on connection failure. A startup connection-timeout check is a reasonable future improvement. |
| **MLflow UI run links not browser-reachable** | Training logs "View run..." links using the internal Docker service address (`http://mlflow:5000`), not reachable from a host browser. Use `http://localhost:5000` directly instead. Cosmetic only. |
| **Every Optuna trial logs a full model artifact** | Storage is not yet optimized — only the winning trial's model is functionally needed. A cleaner approach (params/metrics only for trials, full artifacts only for candidate/production) is a planned refinement. |
| **Per-service image split had limited size impact** | Splitting into `api`/`trainer`/`mlflow` images reduced size only marginally (~1-2%) — TensorFlow itself, shared by `api` and `trainer`, dominates image size far more than the training-only extras (Optuna, yfinance, pandas) that were removed. `tensorflow-cpu` and multi-stage builds are the more impactful next steps if image size matters for cloud deployment cost. |
| **CI builds but does not push or deploy images** | The GitHub Actions pipeline currently validates that all three images build successfully and that unit tests pass. Pushing to a registry and deploying are not yet implemented — see roadmap. |

---

## Phase 2 roadmap

### Hyperparameter search
- [ ] Optuna `suggest_float` / `suggest_int` for continuous search spaces
- [ ] Optuna pruners for early stopping of unpromising trials
- [ ] Walk-forward validation / time series cross-validation
- [ ] Window size as a swept hyperparameter

### Model management
- [ ] Lazy-load `Predictor` (FastAPI dependency, not module-level) for cleaner testability and startup behavior
- [ ] Reduce per-trial artifact logging — params/metrics only during search
- [ ] Mock-based tests for `Predictor` fully replacing the integration suite's import-time coupling

### Image & deployment optimization
- [ ] `tensorflow-cpu` instead of `tensorflow` to reduce image size meaningfully
- [ ] Multi-stage Docker builds to discard pip build artifacts from final layers

### Deployment & monitoring
- [ ] Push built images to a container registry from CI
- [ ] Cloud deployment (AWS/GCP) with automated deploy from CI
- [ ] Prometheus + Grafana monitoring for prediction latency and data drift
- [ ] Returns-based modelling experiment as an alternative to price-level prediction
