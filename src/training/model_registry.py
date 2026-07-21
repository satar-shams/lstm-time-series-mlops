import mlflow
from mlflow import MlflowClient


class ModelRegistry:
    def __init__(self):
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