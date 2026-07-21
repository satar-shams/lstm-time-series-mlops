import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class LSTMEvaluator:
    
    def evaluate_dataset(self, model, scaler: StandardScaler, X: np.ndarray, y_real: np.ndarray) -> dict[str, float]:          
        predictions = model.predict(X, verbose=0)
        y_pred_real = scaler.inverse_transform(predictions)

        mae = mean_absolute_error(
            y_real,
            y_pred_real
        )
        rmse = np.sqrt(
            mean_squared_error(
                y_real,
                y_pred_real
            )
        )
        rmse_percent = (
            rmse / y_real.mean()
        ) * 100

        return {
            "mae": mae,
            "rmse": rmse,
            "rmse_percent": rmse_percent
        }