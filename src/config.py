import os

# ============================================================
# Dataset
# ============================================================

TICKER = "AAPL"

START_DATE = "2010-01-01"
END_DATE = "2026-06-26"

WINDOW_SIZE = 30

TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.95

# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

MLFLOW_MODEL_NAME = "LSTMStockPredictor"

# ============================================================
# Training
# ============================================================

# See MLflow configuration instructions at the end of this file.
# Default: Dockerised MLflow workflow.
MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction Production"

OPTUNA_TRIALS = 25

SEED = 42

# ============================================================
# Default Hyperparameters
# ============================================================

# Every hyperparameter must have a corresponding default value.
# To stop tuning a parameter, remove it from HYPER_PARAMS but keep its default value.
DEFAULTS_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "lstm_units": 64,
    "dense_units": 128,
    "dropout_rate": 0.5,
    "clip_norm": 1.0,
}

# ============================================================
# Hyperparameter Search Space
# ============================================================

# Training stops early if the model stops improving,
# so epochs is just a safe upper limit and isn't tuned.
HYPER_PARAMS = {
    "batch_size": [32, 64],
    "learning_rate": [0.01, 0.001, 0.0001],
    "lstm_units": [32, 64, 128],
    "dense_units": [64, 128, 256],
    "dropout_rate": [0.3, 0.5, 0.7],
    "clip_norm": [0.5, 1.0, 2.0, 5.0],
}


# ============================================================
# MLflow Usage Instructions
# ============================================================

# The project uses a Dockerised MLflow server as the main MLflow
# backend. The experiment name above is created inside this MLflow
# environment.
#
# Recommended workflow:
#
# 1. Start MLflow:
#
#    docker compose up -d mlflow
#
# 2. Run training:
#
#    docker compose run --rm trainer
#
#
# The trainer can also be executed locally while connecting to the
# Dockerised MLflow server:
#
#    source .venv/bin/activate
#
#    python -m src.training.trainer
#
#
# Optional: standalone MLflow server
# ----------------------------------
#
# Running MLflow directly from the host machine is possible for
# development/testing:
#
#    source .venv/bin/activate
#
#    mlflow server \
#      --backend-store-uri sqlite:///mlflow.db \
#      --default-artifact-root ./mlartifacts \
#      --host 0.0.0.0 \
#      --port 5000
#
# Then run the trainer from another terminal:
#
#    source .venv/bin/activate
#
#    python -m src.training.trainer
#
#
# Important:
# ----------
#
# MLflow experiments permanently store their artifact location when
# they are created. Therefore, an experiment created with one MLflow
# storage configuration should not be reused with another one.
#
# Example:
#
# Docker MLflow:
#     MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction Production"
#
# Standalone local MLflow:
#     MLFLOW_EXPERIMENT_NAME = "LSTM Stock Prediction Local"
#
# If switching between different MLflow storage configurations,
# create a separate experiment name.
#
# No automatic environment detection is implemented intentionally.
# The active MLflow environment is selected explicitly through the
# experiment configuration.