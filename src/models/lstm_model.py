from src.config import DEFAULT_LSTM_UNITS, DEFAULT_DENSE_UNITS, DEFAULT_DROPOUT_RATE
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input

class LSTMForecaster:

    def build(self, input_shape : tuple[int, int], lstm_units:int= DEFAULT_LSTM_UNITS, dense_units:int= DEFAULT_DENSE_UNITS, dropout_rate:float= DEFAULT_DROPOUT_RATE) -> Sequential:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(1)
        ])
        return model


