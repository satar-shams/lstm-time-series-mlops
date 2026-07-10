from src.config import HYPER_PARAMS, DEFAULTS_PARAMS
from src.config import WINDOW_SIZE

from src.data.loader import StockLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMForecaster

import numpy as np

from itertools import product
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback

import os

class TimeSeriesTraining:
    def __init__(self):
        pass
    
    # for future training purpose, when we need to change windows, therefore we need new shape of data
    def load_preprocessing_data(self, window_size:int = WINDOW_SIZE) -> dict[str, np.ndarray]:
        stock_loader = StockLoader()            
        dataset = stock_loader.fetch()
        self.preprocess = TimeSeriesPreprocessor(windows_size= window_size)
        preprocessed_data= self.preprocess.run_all(dataset)           
        return preprocessed_data
    
    def prepare_configuration(self) -> list[dict[str, any]]:
        return [
            dict(zip(HYPER_PARAMS.keys(), values))
            for values in product(*HYPER_PARAMS.values())
        ]        
   
    def evaluate_dataset(self, model, X, y_real) -> dict[str, float]:          
        predictions = model.predict(X, verbose=0)
        y_pred_real = self.preprocess.scaler.inverse_transform(predictions)
        
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
    
    def update_best_model(self, metrics:dict[str, float])-> bool:        
        if metrics["rmse_percent"] < self.best_rmse_percent:
            self.best_rmse_percent = metrics["rmse_percent"]
            self.best_rmse = metrics["rmse"]
            self.best_mae = metrics["mae"]
            return True
        else: return False

    def create_callbacks(self)-> list[Callback]:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        return [early_stopping, reduce_lr]

    def show_validation_metrics(self, metrics:dict[str, float]):
        print("\n==============================")
        print("Validation Metrics")
        print("==============================")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"RMSE %: {metrics['rmse_percent']:.2f}%")
        print("==============================")
    
    def show_test_evaluation(self):
        print("X_test shape:", self.data["X_test"].shape)
        print("y_test_real shape:", self.data["y_test_real"].shape)
        metrics  = self.evaluate_dataset(self.best_model, self.data["X_test"], self.data["y_test_real"])
        print("\n==============================")
        print("Final Test Metrics")
        print("==============================")
        print("Best configuration:", self.best_config)
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"RMSE %: {metrics['rmse_percent']:.2f}%")
        print("==============================")

    def save_best_model_scaler(self):
        if self.best_model is None:
            raise RuntimeError("No model was selected — training may have failed for all configs.")        
        os.makedirs("models", exist_ok=True)
        self.best_model.save("models/best_model.keras")
        self.preprocess.save_scaler("models/scaler.bin")
        print("\n✅ Best model saved:")
        print("models/best_model.keras\n")
        # since we need just scaler once for all of models then we save it at the end:
        print("\n✅ scaler saved:")
        print("models/scaler.bin")

    def print_default_values(self)->None:
        print("\nDefault parameters:")
        for key, default_value in DEFAULTS_PARAMS.items():
            if key not in HYPER_PARAMS:
                print(f"  {key} = {default_value}")
        print()

    def train_model(self):
        self.best_model = None
        self.best_config = None
        self.best_rmse = float("inf")
        self.best_rmse_percent = float("inf")
        self.best_mae = float("inf")

        self.data = self.load_preprocessing_data() # we can assign window for future purposes
        configs = self.prepare_configuration()        
        self.print_default_values()
        
        for cfg in configs:
            print("\nTraining with ", end ="")
            for key, value in cfg.items(): print(f"{key} = {value} ", end="")        
            print()

            forecaster = LSTMForecaster()
            model = forecaster.build(input_shape= (self.data["X_train"].shape[1], 1),
                                     lstm_units= cfg.get("lstm_units", DEFAULTS_PARAMS["lstm_units"]),
                                     dense_units= cfg.get("dense_units", DEFAULTS_PARAMS["dense_units"]),
                                     dropout_rate= cfg.get("dropout_rate", DEFAULTS_PARAMS["dropout_rate"])
                                     )
            model.compile(
                optimizer=Adam(
                    learning_rate=cfg.get(
                        "learning_rate",
                        DEFAULTS_PARAMS["learning_rate"],
                    ),
                    clipnorm=cfg.get(
                        "clip_norm",
                        DEFAULTS_PARAMS["clip_norm"],
                    ),
                ),
                loss="mae",
                metrics=[keras.metrics.RootMeanSquaredError()],
            )

            callbacks = self.create_callbacks()
            early_stopping = next(
                cb for cb in callbacks
                if isinstance(cb, EarlyStopping)
            )

            history = model.fit(
                self.data["X_train"], 
                self.data["y_train"],
                validation_data=(
                    self.data["X_val"],
                    self.data["y_val"]
                ),
                epochs=cfg.get("epochs", DEFAULTS_PARAMS["epochs"]), 
                batch_size=cfg.get("batch_size", DEFAULTS_PARAMS["batch_size"]), 
                callbacks=callbacks,
                verbose=0,
            )

            metrics = self.evaluate_dataset(model, self.data["X_val"], self.data["y_val_real"])

            stopped_epoch = len(history.history["loss"])
            best_epoch = early_stopping.best_epoch + 1

            print(f"Training stopped at epoch: {stopped_epoch}")
            print(f"Best epoch: {best_epoch}\n")

            self.show_validation_metrics(metrics)

            if self.update_best_model(metrics):
                self.best_config = cfg
                self.best_model = model

        metrics = self.evaluate_dataset(self.best_model, self.data["X_val"], self.data["y_val_real"])
        
        print("\n Best Model Validation Data")
        self.show_validation_metrics(metrics)
        self.show_test_evaluation()        
        self.save_best_model_scaler()
    
if __name__ == "__main__":
    model = TimeSeriesTraining()
    model.train_model()
 