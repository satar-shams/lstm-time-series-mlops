# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices using
a stacked LSTM model. Built with a focus on engineering rigour: modular OOP
design, Optuna hyperparameter search, MLflow experiment tracking and Model
Registry, three-way train/val/test split, early stopping, adaptive learning
rate scheduling, gradient clipping, reproducible training via fixed seeding,
a fully containerised pipeline with separate images per service, dual model
serving modes (local file or MLflow Registry), structured logging, input
validation, automated tests, continuous integration, and a live cloud
deployment.

> **Status:** Core project complete. Optuna search, MLflow tracking,
> alias-based Model Registry, a Dockerised training/serving pipeline
> (Docker Compose, one image per service), dual model-loading modes
> (local file / MLflow Registry), an automated test suite covering both
> modes, a full CI/CD pipeline (unit tests, container smoke tests,
> automated image publishing to GitHub Container Registry), and a live
> cloud deployment are all complete. Remaining items — validation
> robustness, image size optimisation, monitoring — are tracked as
> optional future work below, not blockers.

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
│   │   ├── model_registry.py       # ModelRegistry — registration, alias-based versioning, model+metadata loading
│   │   ├── callbacks.py            # EarlyStopping + ReduceLROnPlateau factory
│   │   ├── summary.py              # TrainingSummary — structured console output
│   │   └── utils.py                # set_random_seed() — reproducibility across runs
│   └── inference/
│       └── predictor.py            # Predictor — loads model/scaler from local file OR MLflow Registry, predicts
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── example.py                  # Sample prediction payload for Swagger UI
│   ├── api/
│   │   └── routes/
│   │       ├── health.py           # GET /api/v1/health
│   │       └── prediction.py       # POST /api/v1/predict
│   ├── core/
│   │   ├── config.py               # APISettings — env-based host/port/version/MODEL_SOURCE via pydantic-settings
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
│   ├── test_predictor.py           # Predictor unit tests for local and MLflow modes (mocked, skip-conditioned on MODEL_SOURCE)
│   ├── test_predictor_integration.py # Predictor end-to-end integration test (requires MLflow)
│   ├── test_preprocessor.py        # Windowing, three-way split, scaler round-trip
│   └── test_loader.py              # Empty — reserved for future StockLoader tests
├── scripts/
│   └── train.py
├── notebooks/
│   ├── exploration.ipynb
│   ├── LSTM_Training_MLflow.ipynb
│   ├── Load_Save_registered_Model.ipynb
│   └── train_legacy.py             # Original flat training script, kept for reference
├── models/                         # Committed — small trained artifacts (~2.6MB) baked into the API
│   │                                # image for cloud deployment. See "Model loading modes" below.
│   ├── production_model.keras
│   └── scaler.bin
├── mlruns/                         # Gitignored — legacy local MLflow run metadata
├── mlartifacts/                    # Gitignored — MLflow model and scaler artifacts (Docker volume)
├── mlflow.db                       # Gitignored — MLflow SQLite backend (Docker volume)
├── .env                            # Gitignored — local environment variables
├── .env.example                    # Committed template for required environment variables
├── .dockerignore
├── Dockerfile.api                  # API service image — copies models/ into the image, dual-mode capable
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
| Serving | Dual mode — local file (`MODEL_SOURCE=local`) or MLflow Registry (`MODEL_SOURCE=mlflow`), see below |

---

## Model loading modes: local file vs. MLflow Registry

`Predictor` supports two ways to obtain the model and scaler, selected via
the `MODEL_SOURCE` environment variable:

- **`MODEL_SOURCE=local`** — loads `models/production_model.keras` and
  `models/scaler.bin` directly from disk inside the container. No MLflow
  connection required at API startup.
- **`MODEL_SOURCE=mlflow`** — loads the model and scaler from the MLflow
  Model Registry, by alias (`production` by default), the same way earlier
  versions of this project worked exclusively.

**Why both exist:** the MLflow Registry is the intended long-term
architecture — versioned, auditable, swap-a-model-without-rebuilding — and
it is what this project uses for local development and training. However,
running an MLflow server continuously costs more memory than free-tier
cloud hosting typically allows (Render's free tier is capped at 512MB;
MLflow's server process alone approaches or exceeds that even with
`--no-serve-artifacts`). Rather than pay for infrastructure at this stage,
the `local` mode was added so the trained model can be baked directly into
the API image and deployed without a running MLflow server at all.

**Current production deployment (Render) uses `MODEL_SOURCE=local`.** The
MLflow mode is fully implemented and tested locally via Docker Compose, but
is not what is currently serving predictions in the cloud deployment — see
"Cloud deployment" below for the honest, current architecture.

Switching modes requires only a config/env change and rebuilding the `api`
image — no code changes.

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
| Serving | FastAPI + Uvicorn, dual-mode model loading (local file / MLflow Registry) |
| Persistence | joblib (scaler), Keras native `.keras` format (model) |
| Containerisation | Docker + Docker Compose — separate images per service (`api`, `trainer`, `mlflow`) |
| CI/CD | GitHub Actions — unit tests, container smoke test, automated image publishing to GHCR |
| Image registry | GitHub Container Registry (ghcr.io) |
| Cloud hosting | Render — live at `lstm-api-prod.onrender.com` (`MODEL_SOURCE=local`) |
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

### 2. Configure environment variables

```bash
cp .env.example .env
```

`.env.example` contents:

```
APP_HOST=0.0.0.0
APP_PORT=8000

# Model source:
# Options: local | mlflow
MODEL_SOURCE=local
```

`MODEL_SOURCE=local` is the default and matches the current cloud
deployment. Set `MODEL_SOURCE=mlflow` to use the Model Registry instead —
see "Model loading modes" above.

### 3. Build the images

```bash
docker compose build
```

This builds three separate images: `api` (FastAPI + TensorFlow + MLflow
client — inference only), `trainer` (+ Optuna, yfinance, pandas), and
`mlflow` (MLflow only, the smallest of the three).

### 4. Start MLflow and train

```bash
docker compose up -d mlflow
docker compose run --rm trainer
```

This trains the full pipeline: 25-trial Optuna search, candidate model
(tested on the held-out test set), and production model (trained on all
available data). Both candidate and production are registered in the
MLflow Model Registry. The trainer also writes
`models/production_model.keras` and `models/scaler.bin` to the shared
`models/` volume, which is what `MODEL_SOURCE=local` reads from.

Monitor progress in the MLflow UI at `http://localhost:5000`.

> **Note:** the "View run..." links printed during training point at
> `http://mlflow:5000` (the internal Docker service name), not reachable
> from a browser. Use `http://localhost:5000` directly — see Known
> Limitations.

### 5. Start the API

```bash
docker compose up -d api
```

With `MODEL_SOURCE=local` (the default), the API loads
`models/production_model.keras` directly — MLflow does not need to be
running for the API to start. With `MODEL_SOURCE=mlflow`, the API
connects to the MLflow Model Registry instead; `docker-compose.yml`'s
`depends_on` for `mlflow` on the `api` service is commented out by
default since it is only required in `mlflow` mode.

Verify it started successfully:

```bash
docker compose logs api
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
returns a clear `422` error.

### Switching model source

```bash
# .env
MODEL_SOURCE=mlflow
```

then

```bash
docker compose up -d --build api
```

No retraining is required — this only changes where the API loads its
model from, not the model itself.

### Stop everything

```bash
docker compose down
```

---

## Cloud deployment (Render)

The API is currently deployed on Render's free tier, using
`MODEL_SOURCE=local`. The exact sequence used:

1. Train locally via Docker Compose:
   ```bash
   docker compose up -d mlflow
   docker compose run --rm trainer
   ```
   This produces `models/production_model.keras` and `models/scaler.bin`
   on the shared volume.
2. Confirm these files exist in the repository (they are committed —
   see project structure above, ~2.6MB total).
3. Build the API image — `Dockerfile.api` includes `COPY models
   ./models`, baking the trained artifacts directly into the image:
   ```bash
   docker build -f Dockerfile.api -t lstm-api .
   ```
4. Push the image; Render builds and runs it with `MODEL_SOURCE=local`
   (the default), so the API loads the model from disk inside the
   container — no MLflow connection required at runtime.

**Why not MLflow mode in the cloud:** Render's free tier caps memory at
512MB. Running an MLflow server (even with `--no-serve-artifacts` and
`--workers 1`) approaches or exceeds this limit on its own, before the API
itself is accounted for. Rather than pay for larger infrastructure at this
stage, the pipeline was made to support both modes deliberately, and the
cloud deployment uses the mode that has no MLflow memory footprint at
all. Moving the cloud deployment to `MODEL_SOURCE=mlflow` is a possible
future improvement, tracked as optional below.

**Live deployment:** `https://lstm-api-prod.onrender.com`

```bash
curl https://lstm-api-prod.onrender.com/api/v1/health
```

Verified externally (independent of the deployer's own network) via
https://reqbin.com against `/`, `/api/v1/health`, and `/api/v1/predict`
(through `/docs`).

**Corrected architecture — what is actually deployed right now:**

```
Current production mode

Training (local, Docker Compose)
        │
        ▼
  models/ folder
  (production_model.keras, scaler.bin)
        │
        ▼
  API Docker image (COPY models ./models)
        │
        ▼
  Cloud deployment (Render, MODEL_SOURCE=local)


Future / already-implemented-but-not-yet-deployed mode

Training
   │
   ▼
MLflow Model Registry
   │
   ▼
API (MODEL_SOURCE=mlflow)
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
        │               │      MODEL_SOURCE=local:
        │               │      loads models/*.keras
        │               │      baked into image at build
        │               │
        └──────► Model Registry ◄────────┘
                        │        (MODEL_SOURCE=mlflow only)
                        ▼
              MLflow Artifact Store
                (mlartifacts/ volume)
```

Each service is built from its own Dockerfile and its own pinned
requirements file:

- **`mlflow`** (`Dockerfile.mlflow`, `requirements.mlflow.lock`) — the
  central tracking server, model registry, and artifact store. Only
  needed for training and for `MODEL_SOURCE=mlflow` serving.
- **`trainer`** (`Dockerfile.trainer`, `requirements.trainer.lock`) —
  runs once (`docker compose run --rm trainer`), trains and registers
  models to MLflow, and writes the production model/scaler to the shared
  `models/` volume for local-mode serving.
- **`api`** (`Dockerfile.api`, `requirements.api.lock`) — long-running
  FastAPI service. Includes `COPY models ./models` so the image is
  self-contained for `MODEL_SOURCE=local` deployment; also capable of
  `MODEL_SOURCE=mlflow` for local development.

Measuring the per-service split showed only a ~1-2% image size reduction
— TensorFlow itself, shared by `api` and `trainer`, dominates image size
far more than the training-only extras (Optuna, yfinance, pandas) that
were removed. `tensorflow-cpu` and multi-stage builds are the more
impactful next steps if image size matters further.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main`, `dev`, and
`ci-cd-test`, and on pull requests targeting `main`/`dev`:

1. **Unit tests.** Installs dependencies from `requirements.trainer.lock`
   + `requirements.api.lock` + `requirements-dev.txt`, then runs the unit
   test suite with `MODEL_SOURCE=local`. Integration tests requiring
   external services are excluded (`--ignore=tests/test_api.py
   --ignore=tests/test_predictor_integration.py`). `test_predictor.py`'s
   unit tests do **not** require MLflow — they cover both `local` and
   `mlflow` model-loading paths via mocking, with
   `@pytest.mark.skipif(settings.MODEL_SOURCE != ...)` selecting which
   half runs for the current `MODEL_SOURCE`.
2. **Container smoke test.** Builds the `api` image, runs it as a real
   container (`MODEL_SOURCE=local`), waits for startup, then hits
   `/api/v1/health` and `/api/v1/predict` with `curl --fail` against the
   running container — not just a build check, an actual request/response
   round trip. The container is stopped and removed afterward regardless
   of outcome (`if: always()`).
3. **Image publishing.** On push events only (not pull requests), and
   only after the tests and smoke test above pass, all three images
   (`api`, `trainer`, `mlflow`) are built and pushed to GitHub Container
   Registry, tagged both `:latest` and `:<commit-sha>`:
   - `ghcr.io/<owner>/lstm-api`
   - `ghcr.io/<owner>/lstm-trainer`
   - `ghcr.io/<owner>/lstm-mlflow`

Publishing gated behind passing tests and a passing smoke test means a
broken image cannot reach the registry.

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

`test_api.py` and `test_predictor_integration.py` require a running
MLflow service because they instantiate the real application stack,
including a live `Predictor()` connected to the MLflow Model Registry.
They are excluded from CI's unit test run via `--ignore` and executed
separately when the full Docker Compose environment (`mlflow` running,
a registered `production` model) is available.

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
API-specific settings (`MODEL_SOURCE`, `APP_HOST`, `APP_PORT`) live in
`app/core/config.py` via `pydantic-settings`, reading from `.env`.

```python
# src/config.py (excerpt)
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.95
OPTUNA_TRIALS = 25
SEED = 42

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction Production"
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

```python
# app/core/config.py (excerpt)
MODEL_SOURCE_LOCAL = "local"
MODEL_SOURCE_MLFLOW = "mlflow"
MODEL_SOURCE: str  # required — set via .env, no default

MODEL_PATH = "models/production_model.keras"
SCALER_PATH = "models/scaler.bin"
MLFLOW_ALIAS = "production"
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
  model available. When `MODEL_SOURCE=mlflow`, this is the version
  `Predictor` loads; when `MODEL_SOURCE=local`, the same model is loaded
  from `models/production_model.keras` instead, written to disk by the
  same training run that registered it.

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
be enforced at the full dependency-tree level via a lockfile. This lesson
resurfaced during the dual-mode `Predictor` work — an accidental
`from keras.models import load_model` (bypassing `tensorflow.keras`) was
caught and fixed before merging, for exactly this reason.

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
experiment, never editing the old one's stored config.

---

## Known limitations

| Limitation | Notes |
|---|---|
| **Single train/val/test split** | The 90/5/5 split uses one fixed historical window. Walk-forward validation would give a more robust generalization estimate. |
| **Small test set** | 5% of ~4,000 trading days (~200 days) gives a noisier held-out estimate than the earlier 15% split. The current +1.17% generalization gap should be read with this in mind. |
| **Categorical search space only** | Optuna currently uses `suggest_categorical` over discrete lists. `suggest_float`/`suggest_int` with continuous ranges and pruners are not yet implemented. |
| **Test metrics logged (not used) during search** | Every Optuna trial logs test-set metrics for diagnostic visibility, but trial selection strictly uses validation RMSE% only. |
| **Integration tests require external services** | `test_api.py` and `test_predictor_integration.py` require a running MLflow environment and are executed separately from CI's unit test run, not via pytest marker filtering alone. |
| **No multi-step forecasting** | The model predicts one day ahead. Recursive window-sliding for multi-day forecasting is a planned `Predictor` extension. |
| **Placeholder test files empty** | `tests/test_loader.py`, `tests/test_lstm_model.py`, `src/models/base_model.py` are reserved for future work but currently contain no code. |
| **API startup hangs rather than failing fast when MODEL_SOURCE=mlflow and MLflow is unreachable** | `uvicorn` startup does not exit cleanly (may require force-kill). Likely MLflow client retry/backoff. Does not affect `MODEL_SOURCE=local`, which has no MLflow dependency at startup. |
| **MLflow UI run links not browser-reachable** | Training logs "View run..." links using the internal Docker service address (`http://mlflow:5000`). Use `http://localhost:5000` directly instead. Cosmetic only. |
| **Every Optuna trial logs a full model artifact** | Only the winning trial's model is functionally needed; storage is not yet optimized for this. |
| **Per-service image split had limited size impact** | Splitting into `api`/`trainer`/`mlflow` images reduced size only ~1-2% — TensorFlow dominates image size far more than the training-only extras removed. `tensorflow-cpu` and multi-stage builds would be more impactful. |
| **Live cloud deployment uses `MODEL_SOURCE=local`, not MLflow** | Render's free tier (512MB) cannot comfortably run an MLflow server alongside the API. The MLflow-backed serving mode is fully implemented and tested locally via Docker Compose, but the live deployment intentionally uses the mode with no MLflow memory footprint — see "Cloud deployment" above. |
| **Local (non-Docker) training does not currently connect to the Dockerised MLflow instance** | Changes made to support Docker-based artifact storage (`--serve-artifacts`, allowed-hosts) broke the previously-working local `python -m src.training.trainer` path against a locally-run `mlflow server`. Deferred — will be revisited during a full fresh-clone verification pass. Docker-based training (`docker compose run --rm trainer`) is unaffected and is the currently-supported training path. |

---

## Optional future work

The core project (training, tracking, registry, testing, CI/CD, and a live
deployment) is complete. Everything below is a genuine improvement, not a
gap blocking the project's current state.

### Hyperparameter search
- [ ] Optuna `suggest_float` / `suggest_int` for continuous search spaces
- [ ] Optuna pruners for early stopping of unpromising trials
- [ ] Walk-forward validation / time series cross-validation
- [ ] Window size as a swept hyperparameter

### Model management
- [ ] Lazy-load `Predictor` (FastAPI dependency, not module-level)
- [ ] Reduce per-trial artifact logging — params/metrics only during search
- [ ] Fix local (non-Docker) training's connection to Dockerised MLflow
- [ ] Move cloud deployment to `MODEL_SOURCE=mlflow` once infrastructure supports it

### Image & deployment optimization
- [ ] `tensorflow-cpu` instead of `tensorflow` to reduce image size meaningfully
- [ ] Multi-stage Docker builds to discard pip build artifacts from final layers

### Deployment & monitoring
- [ ] Automated deploy from CI directly to Render (image publishing to GHCR is automated; the Render deploy step itself is still manual)
- [ ] Prometheus + Grafana monitoring for prediction latency and data drift
- [ ] Returns-based modelling experiment as an alternative to price-level prediction