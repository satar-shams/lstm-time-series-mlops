import numpy as np

from src.config import WINDOW_SIZE, MLFLOW_MODEL_NAME   
from src.training.model_registry import ModelRegistry

from app.core.exceptions import (
    ModelLoadError,
    PredictionFailedError,
)
class Predictor:
    def __init__(self):
        self.registry = ModelRegistry()
        try:
            self.model, self.scaler = self.registry.load_production_model(
                model_name=MLFLOW_MODEL_NAME,
                alias="production",
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load production model:{e}") from e

    def predict(self, raw_window: list[float]) -> float:
        if len(raw_window) != WINDOW_SIZE:
            raise ValueError(f"Expected {WINDOW_SIZE} values, got {len(raw_window)}")
        try:
            reshaped_raw_data = np.array(raw_window).reshape(-1, 1)

            scaled_data = self.scaler.transform(
                reshaped_raw_data
            )

            input_data = scaled_data.reshape(
                1,
                scaled_data.shape[0],
                1,
            )

            predicted_scaled = self.model.predict(input_data)

            predicted_value = self.scaler.inverse_transform(
                predicted_scaled
            )

            return float(predicted_value[0][0])

        except Exception as e:
            raise PredictionFailedError(
                "Failed to generate prediction"
            ) from e


if __name__ == "__main__":
    predictor = Predictor()
    real_data:list[float] = [276.5751647949219, 283.9184265136719, 287.245361328125, 287.1754150390625, 293.0500183105469, 292.67999267578125, 294.79998779296875, 298.8699951171875, 298.2099914550781, 300.2300109863281, 297.8399963378906, 298.9700012207031, 302.25, 304.989990234375, 308.82000732421875, 308.3299865722656, 310.8500061035156, 312.510009765625, 312.05999755859375, 306.30999755859375, 315.20001220703125, 310.260009765625, 311.2300109863281, 307.3399963378906, 301.5400085449219, 290.54998779296875, 291.5799865722656, 295.6300048828125, 291.1300048828125, 296.4200134277344] # , 299.239990234375, 295.95001220703125, 298.010009765625, 297.010009765625, 294.29998779296875, 293.0799865722656, 275.1499938964844

    real_value = 299.239990234375
    predicted_value =  predictor.predict(real_data)
    print(f"real data : {real_value}" )
    print(f"predicted data : {predicted_value}" )
    print(type(predicted_value))

    # how to get data in loader
    # values = dataset["Close"].tail(31).tolist()
    # print(values)
    # print(len(values))