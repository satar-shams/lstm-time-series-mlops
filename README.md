# LSTM Time Series Forecasting — End-to-End MLOps Project

A production-structured ML pipeline for forecasting AAPL stock prices with a stacked LSTM model — Optuna hyperparameter search, MLflow experiment tracking and Model Registry, a fully containerised training/serving pipeline, automated testing, CI/CD with container smoke tests and automated image publishing, and a **live cloud deployment**.

**Live API:** [`https://lstm-api-prod.onrender.com`](https://lstm-api-prod.onrender.com/) · [Swagger docs](https://lstm-api-prod.onrender.com/docs)
*Temporary deployment — available until approximately mid-September 2026.*


> **Status:** Core project complete — training pipeline, experiment tracking, model registry, dual-mode serving, automated tests, CI/CD, and a live deployment are all implemented and verified. Remaining items (walk-forward validation, image size optimisation, monitoring) are tracked as optional future work — see [Optional future work](#optional-future-work).

---

## Documentation

This README covers the essentials. For deeper detail, see:

| Doc                                            | Covers                                                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md) | Full system architecture, end-to-end pipeline diagram, Docker service breakdown, and complete project structure |
| [`docs/decisions.md`](docs/decisions.md)       | Engineering decisions and the reasoning behind each one                                                         |
| [`docs/deployment.md`](docs/deployment.md)     | Full cloud deployment workflow and the local/MLflow migration path                                              |
| [`docs/mlflow.md`](docs/mlflow.md)             | MLflow tracking, Model Registry, aliasing strategy, and the artifact-location incident                          |
| [`docs/testing.md`](docs/testing.md)           | Testing strategy across unit, integration, and container smoke tests                                            |

---

## System Architecture

The project follows a production-oriented pipeline from data acquisition through
model training, model registry management, API serving, and cloud deployment.

![System Architecture](docs/images/system_architecture.png)

---

## Tech stack

| Layer                 | Tools                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------- |
| Model                 | TensorFlow 2.18.0 / Keras 3.15.0                                                      |
| Data                  | yfinance, pandas, NumPy                                                               |
| Preprocessing         | scikit-learn `StandardScaler`                                                         |
| Hyperparameter search | Optuna 4.9.0 (TPE sampler)                                                            |
| Experiment tracking   | MLflow 3.14.0 (SQLite backend, `--serve-artifacts` proxy, Model Registry)             |
| Serving               | FastAPI + Uvicorn, dual-mode model loading (local file / MLflow Registry)             |
| Persistence           | joblib (scaler), Keras native `.keras` format (model)                                 |
| Containerisation      | Docker + Docker Compose — separate image per service (`api`, `trainer`, `mlflow`)     |
| CI/CD                 | GitHub Actions — unit tests, container smoke test, automated image publishing to GHCR |
| Image registry        | GitHub Container Registry (GHCR)                                                      |
| Cloud hosting         | Render — live at `lstm-api-prod.onrender.com`                                         |
| Tests                 | pytest, httpx                                                                         |

---

## Results

**Full 25-trial Optuna search:**

| Metric                                 |      Value | Meaning                                                                                        |
| -------------------------------------- | ---------: | ---------------------------------------------------------------------------------------------- |
| Optimization RMSE% (validation)        |      1.94% | Used for trial selection — trial 22                                                            |
| Trial's own test RMSE%                 |      1.52% | Diagnostic only — logged per trial, never used for selection                                   |
| **Final test RMSE% (candidate model)** |  **3.11%** | Retrained on train+val, evaluated once on the untouched test set — this is the reported result |
| **Generalization gap**                 | **+1.17%** | Final test RMSE% − optimization RMSE%                                                          |

**Best hyperparameters found:**

| Parameter                  | Value |
| -------------------------- | ----: |
| `batch_size`               |    32 |
| `learning_rate`            | 0.001 |
| `lstm_units`               |   128 |
| `dense_units`              |   256 |
| `dropout_rate`             |   0.3 |
| `clip_norm`                |   5.0 |
| `best_epoch` (from search) |    30 |

The production model is retrained on **all available data** (train + val + test) using this configuration once the honest test score above has been recorded — no further held-out evaluation is possible or needed at that stage.

### Why two test RMSE% numbers appear, and which one counts

Each Optuna trial's own model is evaluated on the test set purely for diagnostic visibility. That score is **never used for hyperparameter selection**.

The number that matters is the **candidate model's** test RMSE%: it is retrained from scratch on train+validation using the winning trial's fixed `best_epoch`, then evaluated on the test set exactly once.

That is the only test-set number used for reporting or for computing the generalization gap above.

### Investigation: closing the validation-test gap

An earlier version of this pipeline used a 70/15/15 split and saw generalization gaps as large as **+19.86%**. A configuration that looked excellent during search performed substantially worse on held-out data.

Two hypotheses were tested in order:

1. **Callback state loss during retraining** — the candidate model is retrained without `EarlyStopping`/`ReduceLROnPlateau`, using only a fixed epoch count. Comparing the original per-trial model against the retrained one on the same test set showed only a small gap (7.48% vs. 6.26%), ruling this out as the primary cause.
2. **Insufficient training data** — with 70% of ~4,000 trading days, the model had comparatively little history. Moving to a 90/5/5 split provided substantially more training data while preserving genuinely held-out evaluation, reducing the gap to +0.89% in initial testing and +1.17% in the full run above.

A smaller test set (5% of history, ~200 trading days) also means a noisier generalization estimate than the earlier 15% split. The improvement is therefore meaningful, but should be interpreted in that context rather than treated as the final word on generalization.

---

## Model

| Property          | Value                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| Architecture      | Input(30,1) → LSTM → LSTM → Dense(relu) → Dropout → Dense(1) — units tuned by Optuna           |
| Input / target    | 30-day rolling window of closing price → next-day closing price                                |
| Split             | 90% train / 5% validation / 5% test, chronological — no shuffling                              |
| Scaler            | `StandardScaler` fit on training data only                                                     |
| Optimizer         | Adam with gradient clipping (`clipnorm`, tuned)                                                |
| Callbacks         | `EarlyStopping` (patience=10) + `ReduceLROnPlateau` (factor=0.5, patience=5)                   |
| Reproducibility   | Fixed seed + `tf.config.experimental.enable_op_determinism()`                                  |
| Training pipeline | Three tiers — search (validation) → candidate (train+val, tested once) → production (all data) |
| Model management  | MLflow Model Registry, alias-based versioning (`candidate` / `production`)                     |

![LSTM Model Architecture](docs/images/lstm_model_architecture.png)
---

## Model serving: local file vs. MLflow Registry

`Predictor` supports two ways to obtain the model and scaler, selected by the `MODEL_SOURCE` environment variable — no application code changes are required to switch:

* **`local`** — current live deployment. Loads `models/production_model.keras` and `models/scaler.bin` directly from the container filesystem. No MLflow connection is required at startup.
* **`mlflow`** — loads from the MLflow Model Registry by alias (`production` by default). This mode is fully implemented and tested locally through Docker Compose.

**Why the live deployment uses `local`:** Render's free tier caps memory at 512 MB, making a continuously running MLflow server alongside the API impractical at that budget. The project therefore supports both modes deliberately, while the live deployment uses the mode with no MLflow runtime dependency.

Full reasoning and the migration path are documented in [`docs/deployment.md`](docs/deployment.md).

---

## Quickstart (Docker)

The training workflow and application services run through Docker Compose. No local Python environment is required.

```bash
git clone https://github.com/satar-shams/lstm-time-series-mlops.git
cd lstm-time-series-mlops
cp .env.example .env
docker compose build
```

### Start MLflow and train

```bash
docker compose up -d mlflow
docker compose run --rm trainer
```

The trainer executes the full pipeline:

```text
Data loading
    ↓
Preprocessing
    ↓
Optuna hyperparameter search
    ↓
Candidate training + test evaluation
    ↓
Production retraining
    ↓
MLflow tracking + Model Registry
```

### Start the API

`MODEL_SOURCE=local` is the default for local serving.

```bash
docker compose up -d api
```

### Test the API

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [276.58, 283.92, 287.25, 287.18, 293.05, 292.68, 294.80, 298.87, 298.21, 300.23, 297.84, 298.97, 302.25, 304.99, 308.82, 308.33, 310.85, 312.51, 312.06, 306.31, 315.20, 310.26, 311.23, 307.34, 301.54, 290.55, 291.58, 295.63, 291.13, 296.42]}'
```

Expected prediction for this example:

```json
{"prediction": 293.33}
```

> Exact predictions may vary after retraining because the model is retrained from scratch. The documented value above is an example from the current project state.

Wrong-length input returns a clear `422` validation response.

Swagger UI:

```text
http://localhost:8000/docs
```

### Switch to MLflow-backed serving

The project already supports MLflow-backed model loading. The only difference is the model source used by the API.

After the initial training and model registration, no retraining or image rebuild is required.

Change `.env`:

```dotenv
MODEL_SOURCE=mlflow
```

Then restart the services:

```bash
docker compose down
```

Start MLflow:

```bash
docker compose up -d mlflow
```

Then start the API:

```bash
docker compose up -d api
```

The API will now load the model from the MLflow Model Registry using the `production` alias.

The API endpoints remain unchanged; use the checks above to verify the deployment.

To switch back to local model loading:

```dotenv
MODEL_SOURCE=local
```

Then restart the API:

```bash
docker compose down
docker compose up -d api
```

For the full architecture, per-service Dockerfiles, and image-splitting rationale, see [`docs/architecture.md`](docs/architecture.md).

---

## Project structure

The repository is intentionally organised around separate ML, serving, testing, and deployment concerns:

```text
lstm-time-series-mlops/
├── src/                    # ML pipeline: data, training, inference
├── app/                    # FastAPI serving application
├── tests/                  # Unit and integration tests
├── docs/                   # Architecture, decisions, deployment, MLflow, testing, Images, Screenshots
├── notebooks/              # MLflow experiments and legacy training reference
├── models/                 # Production model artifacts (~2.6 MB)
├── Dockerfile.api          # API container image
├── Dockerfile.trainer      # Training container image
├── Dockerfile.mlflow       # MLflow container image
├── docker-compose.yml      # Local multi-service orchestration
├── requirements.api.lock   # API dependency lockfile
├── requirements.trainer.lock
├── requirements.mlflow.lock
├── requirements-dev.txt    # Development/test dependencies
└── .github/workflows/      # CI/CD pipeline
```

For the complete annotated tree, see [`docs/architecture.md`](docs/architecture.md).

---

## CI/CD

`.github/workflows/ci.yml` runs on every push to `main`, `dev`, and `ci-cd-test`, and on pull requests targeting `main` or `dev`.

![CI/CD Pipeline](docs/images/ci_cd_pipeline.png)

### 1. Unit tests

Runs the fast test suite with `MODEL_SOURCE=local`, excluding the two tests that require a live MLflow environment:

* `test_api.py`
* `test_predictor_integration.py`

`test_predictor.py` covers **both** serving modes through mocking and `pytest.mark.skipif`.

### 2. Container smoke test

CI builds the real API image, starts it as a container, and sends real HTTP requests to:

* `/api/v1/health`
* `/api/v1/predict`

Requests use `curl --fail`, so this validates an actual container request/response round trip rather than merely checking that the image builds.

### 3. Image publishing

On push events, and only after the test and smoke-test stages pass, CI builds and publishes all three service images to GitHub Container Registry with:

```text
:latest
:<commit-sha>
```

Full testing strategy: [`docs/testing.md`](docs/testing.md).

---

## Run tests

Install the locked runtime dependencies and development dependencies:

```bash
pip install -r requirements.trainer.lock \
            -r requirements.api.lock \
            -r requirements-dev.txt
```

### Unit tests only

This is the fast suite used by CI:

```bash
python -m pytest tests/ -v \
  --ignore=tests/test_api.py \
  --ignore=tests/test_predictor_integration.py
```

### Full suite

Requires MLflow to be running with a registered `production` model:

```bash
python -m pytest tests/ -v
```

For the complete testing strategy, see [`docs/testing.md`](docs/testing.md).

---

## Configuration

Training and model defaults are centralised in `src/config.py`. API settings are managed by `app/core/config.py` using `pydantic-settings` and `.env`.

### Training configuration

```python
TICKER = "AAPL"
WINDOW_SIZE = 30
TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.95
OPTUNA_TRIALS = 25
SEED = 42

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

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

> Every parameter in `HYPER_PARAMS` must have a matching entry in `DEFAULTS_PARAMS`. Removing a parameter from `HYPER_PARAMS` while keeping its default stops that parameter from being tuned without breaking the pipeline.

### API configuration

```python
MODEL_SOURCE: str  # required: "local" or "mlflow"
MODEL_PATH = "models/production_model.keras"
SCALER_PATH = "models/scaler.bin"
MLFLOW_ALIAS = "production"
```

### MLflow experiment naming

An MLflow experiment's artifact location is fixed at creation time. Changing the artifact storage configuration — for example, moving from a standalone MLflow server to Docker — therefore requires a new experiment rather than reusing an old one.

See the [MLflow incident record](docs/mlflow.md#incident-mlflow-artifact-location-after-docker-migration) for the full investigation.

### Model Registry: candidate vs. production

Every training run registers two model versions under `LSTMStockPredictor`:

* **`@candidate`** — trained on train+validation and evaluated once on the held-out test set. This carries the only honest performance number in the pipeline and is not served.
* **`@production`** — trained on train+validation+test using the same proven configuration. It has no held-out metrics of its own and is the version actually served.

---

## Incident record: Keras/TensorFlow version drift

**Symptom:** The model saved successfully but crashed at container startup with:

```text
TypeError: Unrecognized keyword arguments passed to Dense:
{'quantization_config': None}
```

**Root cause:** `tensorflow==2.18.0` does not pin the Keras version. Since TensorFlow 2.16, Keras 3 ships independently. The training environment resolved `keras==3.15.0`, while the Docker image — built from a requirements file that only pinned TensorFlow — resolved a different Keras 3.x release. A newer `Dense` configuration key was therefore not recognised by the older runtime.

**Fix:** Pinned `keras==3.15.0` explicitly, generated complete lockfiles using `pip freeze`, rebuilt the image from those lockfiles, and matched the Dockerfile's Python version to the training environment.

**Lesson:** Pinning a top-level framework does not necessarily pin its complete dependency tree. Train/serve parity must be enforced across the resolved environment, not only direct dependencies.

The same underlying issue resurfaced once during later work (`from keras.models import ...` instead of `tensorflow.keras`) and was caught during review before merging.

Full engineering decision: [`docs/decisions.md`](docs/decisions.md#pin-complete-dependency-environments).

---

## Incident record: stale MLflow artifact location

**Symptom:** After migrating MLflow to Docker with `--serve-artifacts`, training runs logged parameters and metrics correctly, but the Artifacts tab was empty for the pre-existing experiment.

**Root cause:** MLflow fixes an experiment's `artifact_location` permanently at creation time. The experiment had originally been created during local, non-Docker training and therefore pointed to a host filesystem path that did not exist inside the container.

**Fix:** Created a new experiment, `LSTM Stock Prediction Production`, rather than attempting to repair the old experiment. The new experiment picked up the current server's artifact configuration. The fix was first verified independently with a minimal `mlflow.log_artifact()` test before retraining the complete pipeline.

**Lesson:** Experiment metadata must be treated as part of the infrastructure configuration. Storage migrations can invalidate existing artifact locations, so a new experiment may be required when the underlying storage architecture changes.

Full incident record: [`docs/mlflow.md`](docs/mlflow.md#incident-mlflow-artifact-location-after-docker-migration).

---

## Known limitations

| Limitation                                          | Notes                                                                                                                                                                                |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Single train/val/test split**                     | One fixed historical window. Walk-forward validation would provide a more robust generalization estimate.                                                                            |
| **Small test set**                                  | 5% of ~4,000 days (~200) is noisier than the earlier 15% split — interpret the +1.17% gap in that context.                                                                           |
| **Categorical search space only**                   | Optuna currently uses `suggest_categorical` over discrete lists; continuous ranges and pruners are not implemented.                                                                  |
| **Integration tests require external services**     | `test_api.py` and `test_predictor_integration.py` require a running MLflow environment and are excluded from CI's unit-test stage.                                                   |
| **No multi-step forecasting**                       | The model predicts one day ahead only; recursive window-sliding is a planned extension.                                                                                              |
| **MLflow startup dependency**                       | With `MODEL_SOURCE=mlflow`, an unreachable MLflow server can prevent successful model loading at startup. This does not affect the live deployment, which uses `MODEL_SOURCE=local`. |
| **MLflow UI run links**                             | Training logs the internal Docker address (`http://mlflow:5000`); local browser access uses `http://localhost:5000`. Cosmetic only.                                                  |
| **Every Optuna trial logs a full model artifact**   | Only the winning trial's model is ultimately required; artifact storage is not yet optimised.                                                                                        |
| **Per-service image split had limited size impact** | The split produced only ~1–2% reduction because TensorFlow dominates image size. `tensorflow-cpu` and multi-stage builds would have greater impact.                                  |

---

## Optional future work

The core project — training, tracking, registry, serving, testing, CI/CD, and live deployment — is complete. The following are genuine improvements rather than blockers to the current implementation:

* **Validation** — walk-forward/time-series cross-validation, additional metrics beyond RMSE%, window size as a tuned hyperparameter
* **Hyperparameter search** — continuous ranges (`suggest_float` / `suggest_int`), Optuna pruning
* **Serving** — lazy-load `Predictor` through FastAPI dependency injection, model health checks, API rate limiting
* **Deployment** — automated deployment from CI, scheduled retraining, `tensorflow-cpu` and multi-stage builds for image size, managed MLflow hosting for cloud Registry serving
* **Monitoring** — Prometheus/Grafana for latency and service health, data-drift detection, post-deployment performance tracking

For the detailed rationale behind these areas, see [`docs/decisions.md`](docs/decisions.md) and the future-improvement sections in [`docs/architecture.md`](docs/architecture.md) and [`docs/mlflow.md`](docs/mlflow.md).

---

## AI-Assisted Development

AI assistants were used throughout the development of this project as engineering and learning support.

The author implemented the project and remained responsible for its architecture, design decisions, debugging, testing, deployment configuration, and final code. AI assistance was used for tasks such as:

* Explaining unfamiliar technical concepts and technologies.
* Discussing implementation approaches and architecture decisions.
* Reviewing implementation ideas and identifying potential issues.
* Troubleshooting errors during development, Docker, MLflow, CI/CD, and deployment.
* Supporting documentation based on the author's implementation, test results, investigations, and engineering decisions.

The project also influenced a more implementation-first development workflow for subsequent work, where the author focuses on implementing unfamiliar components independently and uses AI primarily for review, debugging, explanation, and design discussion.

Documentation in this repository was developed with AI assistance from the author's own project notes, implementation details, test results, and incident investigations, and was reviewed by the author.

AI assistance was provided primarily through **ChatGPT (OpenAI)** and **Claude (Anthropic)**.

This disclosure is provided for transparency. The author remains responsible for the project's architecture, implementation, engineering decisions, testing, deployment, and final contents of the repository.

---

**LSTM Time Series Forecasting — End-to-End MLOps**
Built and maintained by **Satar Shamsi Goushki**

[GitHub](https://github.com/satar-shams/lstm-time-series-mlops) · [Live API](https://lstm-api-prod.onrender.com/) · [Swagger Docs](https://lstm-api-prod.onrender.com/docs)

*Temporary deployment — available until approximately mid-September 2026.*

© 2026 Satar Shamsi Goushki. All rights reserved.
