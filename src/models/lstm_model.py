from src.config import DEFAULTS_PARAMS
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input

class LSTMForecaster:

    def build(self, input_shape : tuple[int, int], lstm_units:int= DEFAULTS_PARAMS["lstm_units"], dense_units:int= DEFAULTS_PARAMS["dense_units"], dropout_rate:float= DEFAULTS_PARAMS["dropout_rate"]) -> Sequential:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(1)
        ])
        return model