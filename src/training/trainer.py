from src.config import HYPER_PARAMS, DEFAULTS_PARAMS 
from src.config import WINDOW_SIZE, OPTUNA_TRIALS
from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,MLFLOW_MODEL_NAME

from src.data.loader import StockLoader
from src.data.preprocessor import TimeSeriesPreprocessor

import numpy as np

import mlflow
import os

from src.training.evaluator import LSTMEvaluator
from src.training.mlflow_manager import MLFlowManager
from src.training.optuna_tuner import LSTMOptuna
from src.training.single_model_trainer import SingleModelTrainer
from src.training.summary import TrainingSummary
from src.training.model_registry import ModelRegistry

from optuna.study import Study

from src.training.utils import set_random_seed
class TimeSeriesTraining:
    def __init__(self):
        self.evaluator = LSTMEvaluator()
        self.mlflow_manager = MLFlowManager()
        self.optuna_tuner = LSTMOptuna()
        self.trainer = SingleModelTrainer()
        self.summary = TrainingSummary()
        self.registry = ModelRegistry()
         
    # for future training purpose, when we need to change windows, therefore we need new shape of data
    def load_preprocessing_data(self, window_size:int = WINDOW_SIZE) -> dict[str, np.ndarray]:
        stock_loader = StockLoader()            
        dataset = stock_loader.fetch()
        self.preprocess = TimeSeriesPreprocessor(windows_size= window_size)
        preprocessed_data= self.preprocess.run_all(dataset)           
        return preprocessed_data

    def save_model_locally(self, model):
        os.makedirs("models", exist_ok=True)
        model.save("models/production_model.keras")
        print("\n✅ Production model saved in:")
        print("models/production_model.keras\n")

    def prepare_training(self):
        set_random_seed()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        self.data = self.load_preprocessing_data() # we can assign window for future purposes
        os.makedirs("models", exist_ok=True)
        self.preprocess.save_scaler("models/scaler.bin")

        self.summary.default_values(DEFAULTS_PARAMS=DEFAULTS_PARAMS, HYPER_PARAMS=HYPER_PARAMS)

    def get_best_trial_config(self, study: Study):
        best_trial = study.best_trial
        cfg = best_trial.params.copy()
        cfg["epochs"] = best_trial.user_attrs["epochs"]

        best_run_id = best_trial.user_attrs["run_id"]
        print("\nLoaded Best Trial Configuration:", cfg)
        print("Best Trial Run ID: ",best_run_id)

        return cfg, best_run_id

    def evaluate_model(self, model):

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

    def log_candidate_model(
        self,
        model,
        cfg,
        metrics,
        study,
        best_run_id,
    ):
        with mlflow.start_run(run_name="candidate_model") as run:
            self.mlflow_manager.log_param("source_trial_number", study.best_trial.number)
            self.mlflow_manager.log_param("training_dataset", "train+validation")
            self.mlflow_manager.log_param("source_trial_run_id", best_run_id)            
            self.mlflow_manager.log_params(cfg)
            self.mlflow_manager.log_metrics(metrics, "test")           
            self.mlflow_manager.log_metric("best_validation_rmse_percent", study.best_trial.value)            
            self.mlflow_manager.log_model(model, WINDOW_SIZE)
            self.mlflow_manager.log_scaler()
            return run.info.run_id
    
    def log_production_model(self, model, cfg):
            # No metrics logged — there is no held-out data left to evaluate
            # against honestly, since this model was trained on train+val+test.
            # The candidate model's test metrics (logged separately) remain the
            # last honest performance estimate for this configuration.
            with mlflow.start_run(run_name="production_model") as run:
                self.mlflow_manager.log_param("training_dataset", "train+validation+test")
                self.mlflow_manager.log_params(cfg)
                self.mlflow_manager.log_model(model, WINDOW_SIZE)
                self.mlflow_manager.log_scaler()
                return run.info.run_id 

    def run(self):
        self.summary.print_section("Preparing Training")
        self.prepare_training()

        self.summary.print_section("Hyperparameter Optimization")
        print(f"Number of trials : {OPTUNA_TRIALS}")
        print("Direction        : minimize (Validation RMSE %)\n")
        study = self.optuna_tuner.optimize_hyperparameters(data= self.data,
                                                           default_params= DEFAULTS_PARAMS,
                                                           hyper_params= HYPER_PARAMS,
                                                           n_trial= OPTUNA_TRIALS,
                                                           scaler= self.preprocess.scaler,
                                                           window_size= WINDOW_SIZE)

        self.summary.best_trial(study)

        X_candidate = np.concatenate([self.data["X_train"], self.data["X_val"]], axis=0,)
        y_candidate = np.concatenate([self.data["y_train"], self.data["y_val"]], axis=0,)

        # Test data is included here deliberately — evaluation against it already
        # happened (see evaluate_model above). This is the final production model,
        # trained on all available history, with no further held-out evaluation.
        X_production = np.concatenate([X_candidate, self.data["X_test"]], axis=0,)
        y_production = np.concatenate([y_candidate, self.data["y_test"]], axis=0,)
        
        cfg, best_run_id = self.get_best_trial_config(study)

        self.summary.print_section("Training Candidate Model (Train + Validation)")
        candidate_model, _ = self.trainer.train(cfg= cfg, X= X_candidate, y= y_candidate)
        metrics = self.evaluate_model(candidate_model)
        
        candidate_run_id  = self.log_candidate_model(
            candidate_model,
            cfg,
            metrics,
            study,
            best_run_id,
        )
        
        version = self.registry.register_model(
            run_id=candidate_run_id,
            model_name=MLFLOW_MODEL_NAME,
        )

        self.registry.set_alias(
            model_name=MLFLOW_MODEL_NAME,
            version=version,
            alias="candidate",
        )

        self.summary.print_section("Training Production Model (All Data)    ")
        production_model, _ = self.trainer.train(
            cfg=cfg,
            X=X_production,
            y=y_production,
        )

        production_run_id = self.log_production_model(
            production_model,
            cfg,
        )
        
        version = self.registry.register_model(
            run_id=production_run_id,
            model_name=MLFLOW_MODEL_NAME,
        )

        self.registry.set_alias(
            model_name=MLFLOW_MODEL_NAME,
            version=version,
            alias="production",
        )

        self.save_model_locally(production_model)

        self.summary.final_summary(
            study,
            metrics,
            best_run_id,
            version,
        )


if __name__ == "__main__":
    model = TimeSeriesTraining()
    model.run()