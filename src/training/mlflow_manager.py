import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
class MLFlowManager:
    def log_param(self, param_name: str, param_value: int | float | str):
        mlflow.log_param(param_name, param_value)
    
    def log_params(self, cfg: dict[str, int | float | str]):
        for key, value in cfg.items():
            mlflow.log_param(key, value)

    def log_metric(self, metric_name: str, metric_value: int | float):          
        mlflow.log_metric(metric_name, metric_value)

    def log_metrics(self, metrics: dict[str, int | float], metric_prefix: str):
        for key, value in metrics.items():
            mlflow.log_metric(f"{metric_prefix}_{key}", value)

    def log_model(
        self,
        model,
        sequence_length: int
    ):
        signature = ModelSignature(
            inputs=Schema([
                TensorSpec(np.dtype("float32"), (-1, sequence_length, 1))
            ]),
            outputs=Schema([
                TensorSpec(np.dtype("float32"), (-1, 1))
            ]),
        )

        mlflow.tensorflow.log_model(
            model=model,
            name="lstm",
            signature=signature,
        )

    def log_scaler(self):
            mlflow.log_artifact(
                "models/scaler.bin",
                artifact_path="scaler",
            )