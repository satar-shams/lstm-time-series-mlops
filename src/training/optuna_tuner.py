import optuna
from optuna.study import Study
from optuna.trial import Trial
import mlflow
from src.training.callbacks import create_callbacks
from src.training.mlflow_manager import MLFlowManager
from src.training.evaluator import LSTMEvaluator
from src.training.single_model_trainer import SingleModelTrainer

from tensorflow.keras.callbacks import EarlyStopping, Callback

import numpy as np
from sklearn.preprocessing import StandardScaler

class LSTMOptuna:
    def __init__(self):
        self.evaluator = LSTMEvaluator()
        self.mlflow_manager = MLFlowManager()
        self.trainer = SingleModelTrainer()
    
    def optimize_hyperparameters(self, 
                                 n_trial: int, 
                                 data:dict[str, np.ndarray], 
                                 hyper_params: dict[str, int | float], 
                                 default_params: dict[str, int | float],
                                 scaler: StandardScaler,
                                 window_size)-> Study: 
        self.data = data
        self.HYPER_PARAMS = hyper_params
        self.DEFAULT_PARAMS = default_params
        self.WINDOW_SIZE = window_size
        self.scaler = scaler

        study = optuna.create_study(
            direction="minimize",
        )
        study.optimize(
            self.objective,
            n_trials=n_trial,
        )
        return study
    
    def _build_trial_config(self, trial: Trial) -> dict[str, int | float]: 
            return {
                key: (
                    trial.suggest_categorical(key, self.HYPER_PARAMS[key])
                    if key in self.HYPER_PARAMS
                    else self.DEFAULT_PARAMS[key]
                )
                for key in self.DEFAULT_PARAMS
                }
    
    def _log_trial_setup(self, trial: Trial, cfg: dict[str, int | float]) -> None:
            self.mlflow_manager.log_param("trial_number", trial.number)
            self.mlflow_manager.log_params(cfg)

    def _get_early_stopping(self, callbacks: list[Callback]) -> EarlyStopping:
            return next(
                cb for cb in callbacks
                if isinstance(cb, EarlyStopping)
            )  
    
    def _log_trial_results(self, best_epoch: int, stopped_epoch:int, metrics: dict[str, float], model) -> None:
            self.mlflow_manager.log_metric("best_epoch", best_epoch)
            self.mlflow_manager.log_metric("stopped_epoch", stopped_epoch)
            self.mlflow_manager.log_metrics(metrics, "val")
            self.mlflow_manager.log_model(model, self.WINDOW_SIZE) # later hyper parameter window size
  
    def objective(self, trial: Trial) -> float:
        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}"
        ):
            cfg = self._build_trial_config(trial)
            self._log_trial_setup(trial, cfg)
            callbacks = create_callbacks()
            model, history = self.trainer.train(
                  cfg= cfg, 
                  X= self.data["X_train"], 
                  y= self.data["y_train"],
                  validation_data= (self.data["X_val"], self.data["y_val"]),
                  callbacks=  callbacks
                  )

            metrics = self.evaluator.evaluate_dataset(
                model,
                self.scaler,
                self.data["X_val"],
                self.data["y_val_real"],
            )

            early_stopping = self._get_early_stopping(callbacks)
            best_epoch = early_stopping.best_epoch + 1
            stopped_epoch = len(history.history["loss"])
            self._log_trial_results(best_epoch, stopped_epoch, metrics, model)

            return metrics["rmse_percent"]