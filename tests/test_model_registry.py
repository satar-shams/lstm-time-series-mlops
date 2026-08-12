import pytest
from unittest.mock import patch, MagicMock
import mlflow

from src.training.model_registry import ModelRegistry

def test_register_model():
    with patch(
        "src.training.model_registry.mlflow.register_model"
    ) as mock_register:

        mock_register.return_value.version = 3

        registry = ModelRegistry()

        version = registry.register_model(
            run_id="abc123",
            model_name="LSTMStockPredictor",
        )

        assert version == 3

        mock_register.assert_called_once_with(
            model_uri="runs:/abc123/lstm",
            name="LSTMStockPredictor",
        )

def test_set_alias():
    with patch(
        "src.training.model_registry.MlflowClient"
    ) as mock_client:

        registry = ModelRegistry()

        registry.set_alias(
            model_name="LSTMStockPredictor",
            version=3,
            alias="production",
        )

        registry.client.set_registered_model_alias.assert_called_once_with(
            name="LSTMStockPredictor",
            alias="production",
            version=3,
        )

def test_get_model_by_alias_success():
    fake_model_version = MagicMock()

    with patch(
        "src.training.model_registry.MlflowClient"
    ) as mock_client:

        mock_client.return_value.get_model_version_by_alias.return_value = (
            fake_model_version
        )

        registry = ModelRegistry()

        result = registry.get_model_by_alias(
            model_name="LSTMStockPredictor",
            alias="production",
        )

        assert result is fake_model_version

def test_get_model_by_alias_not_found():
    with patch(
        "src.training.model_registry.MlflowClient"
    ) as mock_client:

        mock_client.return_value.get_model_version_by_alias.side_effect = (
            mlflow.exceptions.MlflowException("Alias not found")
        )

        registry = ModelRegistry()

        result = registry.get_model_by_alias(
            model_name="LSTMStockPredictor",
            alias="production",
        )

        assert result is None

def test_load_production_model_success():
    fake_model_version = MagicMock()
    fake_model_version.run_id = "abc123"

    fake_model = MagicMock()
    fake_scaler = MagicMock()

    registry = ModelRegistry()

    with patch.object(
        registry,
        "get_model_by_alias",
        return_value=fake_model_version,
    ), patch.object(
        registry,
        "load_model_by_alias",
        return_value=fake_model,
    ), patch.object(
        registry,
        "load_scaler_by_model_version",
        return_value=fake_scaler,
    ):

        model, scaler, model_info = registry.load_production_model(
            model_name="LSTMStockPredictor",
            alias="production",
        )

        assert model is fake_model
        assert scaler is fake_scaler
        assert isinstance(model_info, dict)
        assert model_info["run_id"] == "abc123"

def test_load_production_model_missing_alias():
    registry = ModelRegistry()

    with patch.object(
        registry,
        "get_model_by_alias",
        return_value=None,
    ):

        with pytest.raises(
            ValueError,
            match="No model found for LSTMStockPredictor@production",
        ):
            registry.load_production_model(
                model_name="LSTMStockPredictor",
                alias="production",
            )