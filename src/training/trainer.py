from src.config import HYPER_PARAMS, DEFAULTS_PARAMS 
from src.config import WINDOW_SIZE, OPTUNA_TRIALS
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

from src.data.loader import StockLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMForecaster

import numpy as np

from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

import optuna

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
        metrics  = self.evaluate_dataset(self.best_model, self.data["X_test"], self.data["y_test_real"])
        print("\n==============================")
        print("Final Test Metrics")
        print("==============================")
        print("Best configuration:", self.best_config)
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"RMSE %: {metrics['rmse_percent']:.2f}%")
        print("==============================")

    def save_best_model(self):
        if self.best_model is None:
            raise RuntimeError("No model was selected — training may have failed for all configs.")        
        os.makedirs("models", exist_ok=True)
        self.best_model.save("models/best_model.keras")
        print("\n✅ Best model saved in:")
        print("models/best_model.keras\n")

    def print_default_values(self)->None:
        print("\nDefault parameters:")

        no_default = True
        for key, default_value in DEFAULTS_PARAMS.items():
            if key not in HYPER_PARAMS:
                print(f"  {key} = {default_value}")
                no_default = False

        if no_default:
            print("  None (all parameters are being tuned)")

        print()

    def load_best_config_from_mlflow(
        self,
        trial_number: int,
    ) -> dict:

        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string=f"params.trial_number = '{trial_number}'",
        )

        if runs.empty:
            raise RuntimeError(
                f"No MLflow run found for trial {trial_number}."
            )

        run = runs.iloc[0]

        cfg = {}

        for key, default_value in DEFAULTS_PARAMS.items():

            value = run[f"params.{key}"]

            cfg[key] = type(default_value)(value)
        
        # Use the best epoch found during Optuna for the final training.
        cfg["epochs"] = int(run["metrics.best_epoch"])

        return cfg

    def objective(self, trial):
        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}"
        ):
            cfg = {
                key: (
                    trial.suggest_categorical(key, HYPER_PARAMS[key])
                    if key in HYPER_PARAMS
                    else DEFAULTS_PARAMS[key]
                )
                for key in DEFAULTS_PARAMS
            }

            mlflow.log_param("trial_number", trial.number)
            for key, value in cfg.items():
                mlflow.log_param(key, value)

            forecaster = LSTMForecaster()
            
            model = forecaster.build(input_shape= (self.data["X_train"].shape[1], 1),
                                    lstm_units= cfg["lstm_units"],
                                    dense_units= cfg["dense_units"],
                                    dropout_rate= cfg["dropout_rate"],
                                    )
            
            model.compile(
                optimizer=Adam(
                    learning_rate=cfg["learning_rate"],
                    clipnorm=cfg["clip_norm"],
                ),
                loss="mae",
                metrics=[keras.metrics.RootMeanSquaredError()],
            )

            callbacks = self.create_callbacks()
            
            history = model.fit(
                self.data["X_train"], 
                self.data["y_train"],
                validation_data=(
                    self.data["X_val"],
                    self.data["y_val"]
                ),
                epochs=cfg["epochs"], 
                batch_size=cfg["batch_size"], 
                callbacks=callbacks,
                verbose=0,
            )

            metrics = self.evaluate_dataset(
                model,
                self.data["X_val"],
                self.data["y_val_real"],
            )

            early_stopping = next(
                cb for cb in callbacks
                if isinstance(cb, EarlyStopping)
            )

            best_epoch = early_stopping.best_epoch + 1
            stopped_epoch = len(history.history["loss"])

            if metrics["rmse_percent"] < self.best_rmse_percent:
                self.best_rmse_percent = metrics["rmse_percent"]
                self.best_epoch = best_epoch

            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("stopped_epoch", stopped_epoch)

            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value)

            signature = ModelSignature(
                inputs=Schema([
                    TensorSpec(np.dtype("float32"), (-1, WINDOW_SIZE, 1))
                ]),
                outputs=Schema([
                    TensorSpec(np.dtype("float32"), (-1, 1))
                ]),
            )

            model_info = mlflow.tensorflow.log_model(
                model=model,
                name="lstm",
                signature=signature,
            )

            mlflow.log_artifact(
                "models/scaler.bin",
                artifact_path="scaler",
            )

            return metrics["rmse_percent"]

    def train_single_model(self, cfg:dict[str, float | int], X:np.ndarray, y:np.ndarray):
        forecaster = LSTMForecaster()
        
        model = forecaster.build(input_shape=(X.shape[1], 1),
                                lstm_units= cfg["lstm_units"],
                                dense_units= cfg["dense_units"],
                                dropout_rate= cfg["dropout_rate"],
                                )
        
        model.compile(
            optimizer=Adam(
                learning_rate=cfg["learning_rate"],
                clipnorm=cfg["clip_norm"],
            ),
            loss="mae",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        
        history = model.fit(
            X, 
            y,
            epochs=cfg["epochs"], 
            batch_size=cfg["batch_size"], 
            verbose=0,
        )

        return model, history

    def train_model(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # mlflow.tensorflow.autolog(disable=True)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        self.data = self.load_preprocessing_data() # we can assign window for future purposes
        os.makedirs("models", exist_ok=True)
        self.preprocess.save_scaler("models/scaler.bin")

        self.print_default_values()        
        
        study = optuna.create_study(
            direction="minimize",
        )
        self.best_rmse_percent = float("inf")
        self.best_epoch = 50

        study.optimize(
            self.objective,
            n_trials=OPTUNA_TRIALS,
        )

        print("\nBest trial:")
        print("\nBest trial")
        print(f"Trial: {study.best_trial.number}")
        print(f"Validation RMSE %: {study.best_trial.value:.4f}")

        print("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        X_train_final = np.concatenate(
            [self.data["X_train"], self.data["X_val"]],
            axis=0,
        )

        y_train_final = np.concatenate(
            [self.data["y_train"], self.data["y_val"]],
            axis=0,
        )

        cfg = self.load_best_config_from_mlflow(
                study.best_trial.number
            )
        
        print("\nLoaded config from MLflow:", cfg)

        model, history = self.train_single_model(cfg= cfg,
                                                 X= X_train_final,
                                                 y= y_train_final)
        metrics = self.evaluate_dataset(
            model,
            self.data["X_test"],
            self.data["y_test_real"],
        )

        print("\nFinal Test Metrics")
        print(metrics)

        with mlflow.start_run(run_name="final_model"):
            mlflow.log_param("optuna_trial", study.best_trial.number)
            mlflow.log_param("training_dataset", "train+validation")

            for key, value in cfg.items():
                mlflow.log_param(key, value)

            for key, value in metrics.items():
                mlflow.log_metric(f"test_{key}", value)
           
            mlflow.log_metric(
                "best_validation_rmse_percent",
                study.best_trial.value,
            )
            
            signature = ModelSignature(
                inputs=Schema([
                    TensorSpec(np.dtype("float32"), (-1, WINDOW_SIZE, 1))
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

            mlflow.log_artifact(
                "models/scaler.bin",
                artifact_path="scaler",
            )

    
if __name__ == "__main__":
    model = TimeSeriesTraining()
    model.train_model()
 