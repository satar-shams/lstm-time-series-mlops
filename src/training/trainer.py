from src.config import HYPER_PARAMS, DEFAULTS_PARAMS 
from src.config import WINDOW_SIZE, OPTUNA_TRIALS
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

from src.data.loader import StockLoader
from src.data.preprocessor import TimeSeriesPreprocessor

import numpy as np

import mlflow
import os

from src.training.evaluator import LSTMEvaluator
from src.training.mlflow_manager import MLFlowManager
from src.training.optuna_tuner import LSTMOptuna
from src.training.single_model_trainer import SingleModelTrainer

from optuna.study import Study

from src.training.utils import set_random_seed
class TimeSeriesTraining:
    def __init__(self):
        self.evaluator = LSTMEvaluator()
        self.mlflow_manager = MLFlowManager()
        self.optuna_tuner = LSTMOptuna()
        self.trainer = SingleModelTrainer()
         
    # for future training purpose, when we need to change windows, therefore we need new shape of data
    def load_preprocessing_data(self, window_size:int = WINDOW_SIZE) -> dict[str, np.ndarray]:
        stock_loader = StockLoader()            
        dataset = stock_loader.fetch()
        self.preprocess = TimeSeriesPreprocessor(windows_size= window_size)
        preprocessed_data= self.preprocess.run_all(dataset)           
        return preprocessed_data

    def save_model_locally(self, model):
        os.makedirs("models", exist_ok=True)
        model.save("models/best_model.keras")
        print("\n✅ Best model saved in:")
        print("models/best_model.keras\n")

    def print_section(self, title: str) -> None:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

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

    def print_training_summary(
        self,
        study: Study,
        cfg: dict[str, int | float],
        metrics: dict[str, float],
        best_run_id: str,
    ) -> None:

        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)

        print(f"Best Trial            : {study.best_trial.number}")
        print(f"Validation RMSE (%)   : {study.best_trial.value:.4f}")
        print(f"Final Test RMSE (%)   : {metrics['rmse_percent']:.4f}")
        print(
            f"Generalization Gap    : "
            f"{metrics['rmse_percent'] - study.best_trial.value:+.4f}"
        )

        print("\nArtifacts")
        print("-" * 60)
        print("Model                 : models/best_model.keras")
        print("Scaler                : models/scaler.bin")

        print("\nMLflow")
        print("-" * 60)
        print(f"Best Trial Run ID     : {best_run_id}")

        print("=" * 60)
        print("Training Completed Successfully")
        print("=" * 60)
    
    def prepare_training(self):
        set_random_seed()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        self.data = self.load_preprocessing_data() # we can assign window for future purposes
        os.makedirs("models", exist_ok=True)
        self.preprocess.save_scaler("models/scaler.bin")

        self.print_default_values()

    def show_best_trial(self, study: Study) -> None:
        print("\nBest Trial")
        print("-" * 40)
        print(f"Trial Number         : {study.best_trial.number}")
        print(f"Validation RMSE (%)  : {study.best_trial.value:.4f}")

        print("\nHyperparameters")
        print("-" * 40)
        for key, value in study.best_trial.params.items():
            print(f"{key:<20}: {value}")

    def retrain_best_model(self):
        X_train_final = np.concatenate(
            [self.data["X_train"], self.data["X_val"]],
            axis=0,
        )

        y_train_final = np.concatenate(
            [self.data["y_train"], self.data["y_val"]],
            axis=0,
        )

        cfg, best_run_id = self.mlflow_manager.load_best_config_from_mlflow(
            DEFAULTS_PARAMS= DEFAULTS_PARAMS, 
            MLFLOW_EXPERIMENT_NAME= MLFLOW_EXPERIMENT_NAME
        )
        
        print("\nLoaded config from MLflow:", cfg)
        print("best trial number: ",best_run_id)
        
        model, _ = self.trainer.train(cfg= cfg, X= X_train_final, y= y_train_final)
        
        return model, cfg, best_run_id

    def evaluate_final_model(self, model):

        metrics = self.evaluator.evaluate_dataset(
            model,
            self.preprocess.scaler,
            self.data["X_test"],
            self.data["y_test_real"],
        )

        print("\nTest Metrics")
        print("-" * 40)
        print(f"MAE                 : {metrics['mae']:.4f}")
        print(f"RMSE                : {metrics['rmse']:.4f}")
        print(f"RMSE (%)            : {metrics['rmse_percent']:.4f}")

        return metrics

    def log_final_model(
        self,
        model,
        cfg,
        metrics,
        study,
        best_run_id,
    ):
        with mlflow.start_run(run_name="final_model"):
            self.mlflow_manager.log_param("source_trial_number", study.best_trial.number)
            self.mlflow_manager.log_param("training_dataset", "train+validation")
            self.mlflow_manager.log_param("source_trial_run_id", best_run_id)            
            self.mlflow_manager.log_params(cfg)
            self.mlflow_manager.log_metrics(metrics, "test")           
            self.mlflow_manager.log_metric("best_validation_rmse_percent", study.best_trial.value)            
            self.mlflow_manager.log_model(model, WINDOW_SIZE)
            self.mlflow_manager.log_scaler()
    
    def run(self):
        self.print_section("Preparing Training")
        self.prepare_training()

        self.print_section("Hyperparameter Optimization")
        print(f"Number of trials : {OPTUNA_TRIALS}")
        print("Direction        : minimize (Validation RMSE %)\n")
        study = self.optuna_tuner.optimize_hyperparameters(data= self.data,
                                                           default_params= DEFAULTS_PARAMS,
                                                           hyper_params= HYPER_PARAMS,
                                                           n_trial= OPTUNA_TRIALS,
                                                           scaler= self.preprocess.scaler,
                                                           window_size= WINDOW_SIZE)

        self.show_best_trial(study)

        self.print_section("Retraining Best Model")
        model, cfg, best_run_id = self.retrain_best_model()

        self.save_model_locally(model)

        self.print_section("Final Evaluation")
        metrics = self.evaluate_final_model(model)

        self.log_final_model(
            model,
            cfg,
            metrics,
            study,
            best_run_id,
        )

        self.print_training_summary(
            study,
            cfg,
            metrics,
            best_run_id,
        )

if __name__ == "__main__":
    model = TimeSeriesTraining()
    model.run()