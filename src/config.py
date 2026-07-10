APP_HOST = "127.0.0.1"
APP_PORT = 8000

MODEL_PATH = "models/best_model.keras"
SCALER_PATH = "models/scaler.bin"

TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = "2026-06-26"
WINDOW_SIZE = 30

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.85

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
    # "epochs": [20, 30, 40],
    "batch_size": [32, 64],
    # "learning_rate": [0.01, 0.001, 0.0001],
    # "lstm_units": [32, 64, 128],
    # "dense_units": [64, 128, 256],
    # "dropout_rate": [0.3, 0.5, 0.7],
    # "clip_norm":[0.5, 1.0, 2.0, 5.0],
}
