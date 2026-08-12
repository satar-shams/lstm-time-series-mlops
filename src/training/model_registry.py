import mlflow
from mlflow import MlflowClient
import joblib

from src.config import MLFLOW_TRACKING_URI
class ModelRegistry:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str):
        model_uri = f"runs:/{run_id}/lstm"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        return registered_model.version

    def set_alias(self, model_name: str, version: int, alias: str):
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )

    def get_model_by_alias(self, model_name: str, alias: str = "production"):
        try:
            return self.client.get_model_version_by_alias(model_name, alias)
        except mlflow.exceptions.MlflowException:
            return None

    def get_latest_version_by_alias(self, model_name: str, alias: str = "production"):
        mv = self.get_model_by_alias(model_name, alias)
        return mv.version if mv else None

    def load_model_by_alias(
        self,
        model_name: str,
        alias: str = "production",
    ):
        model_uri = f"models:/{model_name}@{alias}"

        return mlflow.tensorflow.load_model(model_uri)
    
    def load_scaler_by_model_version(
        self,
        model_version,
    ):
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=model_version.run_id,
            artifact_path="scaler/scaler.bin",
        )

        return joblib.load(scaler_path)
        
    def load_production_model(
        self,
        model_name: str,
        alias: str = "production",
    ):
        model_version = self.get_model_by_alias(
            model_name=model_name,
            alias=alias,
        )

        if model_version is None:
            raise ValueError(
                f"No model found for {model_name}@{alias}"
            )

        model = self.load_model_by_alias(
            model_name=model_name,
            alias=alias,
        )

        scaler = self.load_scaler_by_model_version(
            model_version=model_version,
        )

        model_info = {
            "name": model_version.name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "alias": alias,
        }

        return model, scaler, model_info