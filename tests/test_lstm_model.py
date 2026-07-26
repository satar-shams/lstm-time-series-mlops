import numpy as np
from src.models.lstm_model import LSTMForecaster
import pytest

def test_lstm_model_build():
    forecaster = LSTMForecaster()

    model = forecaster.build(
        input_shape=(30, 1),
    )

    assert model.input_shape == (None, 30, 1)
    assert model.output_shape == (None, 1)

def test_lstm_model_architecture():
    forecaster = LSTMForecaster()

    model = forecaster.build(
        input_shape=(30, 1),
    )

    assert len(model.layers)

    assert model.layers[0].__class__.__name__ == "LSTM"
    assert model.layers[1].__class__.__name__ == "LSTM"
    assert model.layers[2].__class__.__name__ == "Dense"
    assert model.layers[3].__class__.__name__ == "Dropout"
    assert model.layers[4].__class__.__name__ == "Dense"

def test_lstm_model_prediction_shape():
    forecaster = LSTMForecaster()

    model = forecaster.build(
        input_shape=(30, 1),
    )

    sample_input = np.random.rand(1, 30, 1).astype("float32")
    prediction = model.predict(
        sample_input,
        verbose=0,
    )
    assert prediction.shape == (1, 1)

def test_lstm_model_compile():
    forecaster = LSTMForecaster()

    model = forecaster.build(
        input_shape=(30, 1),
    )

    forecaster.compile_model(
        model=model,
        learning_rate=0.001,
        clip_norm=5.0,
    )

    assert model.optimizer is not None
    assert model.optimizer.__class__.__name__ == "Adam"
    assert model.loss == "mae"
    assert model.optimizer.learning_rate.numpy() == pytest.approx(0.001)
    assert model.optimizer.clipnorm == pytest.approx(5.0)