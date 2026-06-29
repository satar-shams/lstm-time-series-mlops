TICKER = "AAPL"
START_DATE = "2010-01-01"
END_DATE = "2026-06-26"
WINDOW_SIZE = 30
SPLIT_SIZE = 0.9

DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LSTM_UNITS = 64
DEFAULT_DENSE_UNITS = 128
DEFAULT_DROPOUT_RATE = 0.5

HYPER_PARAMS = {
    "epochs": [20, 30, 40],
    "batch_size": [32, 64],
    # "learning_rate": [0.01, 0.001, 0.0001],
    # "lstm_units": [32, 64, 128],
    # "dense_units": [64, 128, 256],
    # "dropout_rate": [0.3, 0.5, 0.7],
}
