# Testing Strategy

## Overview

The project uses automated tests to validate the reliability of the ML pipeline,
inference components, API behaviour, and deployment artifact.

Testing is organized into three layers:

1. **Unit tests** — validate individual components using mocks where external
   infrastructure would otherwise be required.
2. **Integration tests** — validate selected components against the real
   application or MLflow infrastructure when explicitly run.
3. **Container smoke tests** — validate the actual Docker API image and its
   runtime HTTP behaviour in CI.

The goal is to verify not only individual components, but also the behaviour of
the system as it moves toward deployment.

---

## Testing Architecture

```text
                         Test Suite
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        Unit Tests      Integration Tests   Smoke Tests
         mocked         real application    real Docker
        components      / MLflow when run     container
             │                │                │
             └────────────────┴────────────────┘
                              │
                              ▼
                    CI validation pipeline
```

The regular CI test suite does **not** require a running MLflow server. Live
integration tests are kept separate from the fast CI path.

---

## Unit Tests

Unit tests validate individual components without requiring external services.

External dependencies such as `yfinance`, MLflow Registry operations, and
model loading are mocked where appropriate.

### Data Pipeline

`StockLoader` tests cover:

* Successful data loading.
* Empty responses and error handling.
* MultiIndex column flattening.

Market-data requests are mocked, so tests do not depend on live Yahoo Finance
availability or rate limits.

`TimeSeriesPreprocessor` tests cover:

* Chronological train/validation/test splitting.
* Rolling-window generation.
* Expected input and target shapes.
* Scaler fitting and transformation.
* Inverse-transformation consistency.

These tests protect important time-series properties such as chronological
ordering and preventing accidental data leakage.

### Model

`LSTMForecaster` tests validate:

* Model construction.
* Layer types and ordering.
* Input and output shapes.
* Model compilation.
* Adam optimizer configuration.
* Learning rate.
* Gradient clipping.

These tests detect accidental changes to the model architecture or training
configuration.

### Model Registry

`ModelRegistry` tests use mocked MLflow interactions.

They cover:

* Model registration.
* Alias assignment.
* Model lookup by alias.
* Missing aliases.
* Model loading.
* Scaler and model metadata handling.

This validates the application's Registry logic without requiring an MLflow
server.

### Predictor

`Predictor` tests cover both supported model-loading modes through mocking.

#### Local mode

```text
MODEL_SOURCE=local
```

Tests cover:

* Successful local model loading.
* Scaler loading.
* Successful prediction.
* Model-loading failures.
* Prediction failures.

#### MLflow mode

```text
MODEL_SOURCE=mlflow
```

Tests mock `ModelRegistry.load_production_model()` and cover:

* Successful Registry-based model loading.
* Model-loading failures.
* Prediction failures.

Therefore, these tests verify both inference paths without requiring MLflow
infrastructure.

---

## FastAPI Tests

`tests/test_api.py` exercises the FastAPI application through FastAPI's
`TestClient`.

The tests cover:

* `GET /api/v1/health`
* Successful `POST /api/v1/predict`
* Invalid input length below the required window size.
* Invalid input length above the required window size.
* HTTP status codes and response structure.

These tests are marked as integration tests because they exercise the real
application layer rather than isolated functions.

They are **not part of the fast CI test suite**.

---

## Live Integration Test

The repository also contains
`tests/test_predictor_integration.py`.

Unlike the mocked Predictor tests, this test creates a real `Predictor` and
performs a real inference request.

It requires:

* A running MLflow server.
* A registered `LSTMStockPredictor` model.
* A `production` alias pointing to that model.

The inference path is therefore:

```text
Predictor
    │
    ▼
MLflow Model Registry
    │
    ▼
production alias
    │
    ▼
Registered Model
    │
    ▼
Prediction
```

This test is intentionally separate from the fast CI suite because it requires
external MLflow infrastructure.

---

## Container Smoke Test

CI performs an additional smoke test against the **actual built API Docker
image**.

The workflow:

```text
Build API Image
       │
       ▼
Start Container
       │
       ▼
Wait for Startup
       │
       ▼
GET /api/v1/health
       │
       ▼
POST /api/v1/predict
```

The container runs with:

```text
MODEL_SOURCE=local
```

The smoke test sends real HTTP requests using `curl`.

It therefore validates:

* Docker image construction.
* Dependency availability.
* Model and scaler packaging.
* Container startup.
* Model loading.
* FastAPI runtime behaviour.
* Real HTTP request/response handling.

This is different from the Python-level tests because it validates the actual
deployment artifact.

---

## CI Testing Workflow

GitHub Actions runs the fast automated validation on pushes and pull requests.

The CI workflow:

1. Installs the pinned trainer, API, and development dependencies.
2. Runs the unit test suite with local model configuration.
3. Excludes `test_api.py` and `test_predictor_integration.py` from the fast
   suite.
4. Builds the API Docker image.
5. Starts the built image as a real container.
6. Tests the health endpoint.
7. Tests the prediction endpoint.
8. Stops and removes the test container.
9. On push events, publishes the API, trainer, and MLflow images after the
   validation stages pass.

The CI test suite therefore does **not** require a live MLflow server.

---

## Running Tests Locally

Install the required dependencies:

```bash
pip install -r requirements.trainer.lock \
            -r requirements.api.lock \
            -r requirements-dev.txt
```

### Fast test suite

This is the same basic test suite used by CI:

```bash
python -m pytest tests/ -v \
  --ignore=tests/test_api.py \
  --ignore=tests/test_predictor_integration.py
```

It uses:

```text
MODEL_SOURCE=local
```

and does not require a running MLflow server.

### FastAPI integration tests

The FastAPI tests can be run separately:

```bash
python -m pytest tests/test_api.py -v
```

These tests exercise the real FastAPI application and therefore depend on the
application's configured local inference environment.

### MLflow integration test

The live Predictor integration test requires MLflow and a registered
production model.

The environment can be prepared with:

```bash
docker compose up -d mlflow
docker compose run --rm trainer
```

Then run:

```bash
python -m pytest tests/test_predictor_integration.py -v
```

This test is intentionally separate from the normal fast test suite.

---

## Testing Principles

### Isolation

Components should be testable without requiring the complete infrastructure.

Examples include:

* Mocked MLflow Registry operations.
* Mocked market-data requests.
* Mocked local model loading.
* Mocked MLflow model loading in Predictor tests.

### Reproducibility

Tests should behave consistently across environments through:

* Pinned dependencies.
* Docker-based execution.
* Controlled test inputs.
* Mocked external services where appropriate.

### Deployment Validation

Unit tests alone cannot guarantee that the deployment artifact works.

The CI smoke test therefore validates the actual Docker image, including
startup, model loading, and real HTTP communication.

---

## Future Improvements

Potential future improvements include:

* Replace command-based test exclusions with `pytest` markers.
* Add the live integration tests to a dedicated CI integration job when
  persistent MLflow test infrastructure is available.
* Add model-performance regression tests before model promotion.
* Add automated data-quality validation.
* Add API load and performance testing.
* Add post-deployment verification tests.
