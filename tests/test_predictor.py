from unittest.mock import patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import (
    ModelLoadError,
    PredictionFailedError,
)
from src.inference.predictor import Predictor


class FakeScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class FakeModel:
    def predict(self, data):
        return np.array([[123.45]])


class FailingModel:
    def predict(self, data):
        raise RuntimeError("Model prediction failed")


@pytest.fixture
def sample_input_length() -> list[float]:
    return [
        298.8699951171875,
        298.2099914550781,
        300.2300109863281,
        297.8399963378906,
        298.9700012207031,
        302.25,
        304.989990234375,
        308.82000732421875,
        308.3299865722656,
        310.8500061035156,
        312.510009765625,
        312.05999755859375,
        306.30999755859375,
        315.20001220703125,
        310.260009765625,
        311.2300109863281,
        307.3399963378906,
        301.5400085449219,
        290.54998779296875,
        291.5799865722656,
        295.6300048828125,
        291.1300048828125,
        296.4200134277344,
        299.239990234375,
        295.95001220703125,
        298.010009765625,
        297.010009765625,
        294.29998779296875,
        293.0799865722656,
        275.1499938964844,
    ]


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_LOCAL,
    reason="Local model loading test only",
)
def test_predictor_success(sample_input_length):

    with patch(
        "src.inference.predictor.load_model",
        return_value=FakeModel(),
    ), patch(
        "src.inference.predictor.joblib.load",
        return_value=FakeScaler(),
    ):
        predictor = Predictor()
        prediction = predictor.predict(sample_input_length)

    assert isinstance(prediction, float)


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_LOCAL,
    reason="Local model loading test only",
)
def test_predictor_model_load_failure():

    with patch(
        "src.inference.predictor.load_model",
        side_effect=Exception("Local model unavailable"),
    ):
        with pytest.raises(
            ModelLoadError,
            match="Failed to load local model",
        ):
            Predictor()


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_LOCAL,
    reason="Local model loading test only",
)
def test_predictor_prediction_failure(sample_input_length):

    with patch(
        "src.inference.predictor.load_model",
        return_value=FailingModel(),
    ), patch(
        "src.inference.predictor.joblib.load",
        return_value=FakeScaler(),
    ):
        predictor = Predictor()

        with pytest.raises(
            PredictionFailedError,
            match="Failed to generate prediction",
        ):
            predictor.predict(sample_input_length)


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_MLFLOW,
    reason="MLflow model loading test only",
)
def test_predictor_mlflow_success(sample_input_length):

    fake_model = FakeModel()
    fake_scaler = FakeScaler()

    fake_model_info = {
        "name": "LSTMStockPredictor",
        "version": "20",
        "alias": "production",
        "run_id": "abc123",
    }

    with patch(
        "src.inference.predictor.ModelRegistry.load_production_model",
        return_value=(
            fake_model,
            fake_scaler,
            fake_model_info,
        ),
    ):
        predictor = Predictor()
        prediction = predictor.predict(sample_input_length)

    assert isinstance(prediction, float)


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_MLFLOW,
    reason="MLflow model loading test only",
)
def test_predictor_mlflow_model_load_failure():

    with patch(
        "src.inference.predictor.ModelRegistry.load_production_model",
        side_effect=Exception("MLflow unavailable"),
    ):
        with pytest.raises(
            ModelLoadError,
            match="Failed to load MLflow model",
        ):
            Predictor()


@pytest.mark.skipif(
    settings.MODEL_SOURCE != settings.MODEL_SOURCE_MLFLOW,
    reason="MLflow model loading test only",
)
def test_predictor_mlflow_prediction_failure(sample_input_length):

    fake_model_info = {
        "name": "LSTMStockPredictor",
        "version": "20",
        "alias": "production",
        "run_id": "abc123",
    }

    with patch(
        "src.inference.predictor.ModelRegistry.load_production_model",
        return_value=(
            FailingModel(),
            FakeScaler(),
            fake_model_info,
        ),
    ):
        predictor = Predictor()

        with pytest.raises(
            PredictionFailedError,
            match="Failed to generate prediction",
        ):
            predictor.predict(sample_input_length)