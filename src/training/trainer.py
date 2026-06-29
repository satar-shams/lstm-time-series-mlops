from src.config import HYPER_PARAMS
from src.config import WINDOW_SIZE, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_LSTM_UNITS, DEFAULT_DENSE_UNITS, DEFAULT_DROPOUT_RATE

from src.data.loader import StockLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMForecaster

import numpy as np

from itertools import product
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

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
        TRAINING_CONFIGS = [
        dict(zip(HYPER_PARAMS.keys(), values))
        for values in product(*HYPER_PARAMS.values())
        ]
        return TRAINING_CONFIGS

    def evaluation_best_model(self, model) -> bool:            
        print("ended\n")
        predictions = model.predict(self.data["X_test"], verbose=0)

        y_pred_real = self.preprocess.scaler.inverse_transform(predictions)

        rmse = np.sqrt(
            mean_squared_error(
                self.data["y_test_real"],
                y_pred_real
            )
        )

        rmse_percent = (
            rmse / self.data["y_test_real"].mean()
        ) * 100

        print(f"RMSE %: {rmse_percent:.2f}%")

        if rmse_percent < self.best_rmse_percent:
            self.best_rmse_percent = rmse_percent
            self.best_model = model
            return True
        else: return False
            
    def show_best_model(self):
        print("\n==============================")
        print("Best configuration:", self.best_config)
        print(f"Best RMSE %: {self.best_rmse_percent:.2f}%")
        print("==============================")

    def save_best_model_scaler(self):
        os.makedirs("models", exist_ok=True)

        self.best_model.save("models/best_model.keras")
        self.preprocess.save_scaler("models/scaler.bin")

        print("\n✅ Best model saved:")
        print("models/best_model.keras\n")

        # since we need just scaler once for all of models then we save it at the end:
        print("\n✅ scaler saved:")
        print("models/scaler.bin")
    
    def train_model(self):
        self.best_model = None
        self.best_config = None
        self.best_rmse_percent = float("inf")

        self.data = self.load_preprocessing_data() # we can assign window for future purposes
        configs = self.prepare_configuration()

        # here we can make advance much more flexible and use all or most of parameters default or from cfg
        for cfg in configs:
            print("\nTraining with ", end ="")
            for key, value in cfg.items(): print(f"{key} = {value} ", end="")        
            print()

            forecaster = LSTMForecaster()
            model = forecaster.build(input_shape= (self.data["X_train"].shape[1], 1),
                                     lstm_units= cfg.get("lstm_units", DEFAULT_LSTM_UNITS),
                                     dense_units= cfg.get("dense_units", DEFAULT_DENSE_UNITS),
                                     dropout_rate= cfg.get("dropout_rate", DEFAULT_DROPOUT_RATE)
                                     )

            model.compile(
                optimizer=Adam(cfg.get("learning_rate", DEFAULT_LEARNING_RATE)),
                loss="mae",
                metrics=[keras.metrics.RootMeanSquaredError()]
            )

            model.fit(self.data["X_train"], self.data["y_train"],
                      epochs=cfg.get("epochs", DEFAULT_EPOCHS), batch_size=cfg.get("batch_size", DEFAULT_BATCH_SIZE), verbose=0
            )

            if(self.evaluation_best_model(model)):
                self.best_config = cfg
        
        self.show_best_model()
        self.save_best_model_scaler()

    
if __name__ == "__main__":
    model = TimeSeriesTraining()
    model.train_model()
 