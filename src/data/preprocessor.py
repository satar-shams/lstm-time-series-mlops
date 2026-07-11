import pandas as pd
import numpy as np
import joblib
from src.config import WINDOW_SIZE, TRAIN_SPLIT, VALIDATION_SPLIT

from sklearn.preprocessing import StandardScaler
from src.data.loader import StockLoader

import os
class TimeSeriesPreprocessor:
    def __init__(self,
                 windows_size:int = WINDOW_SIZE,
                 train_split_size:float = TRAIN_SPLIT,
                 validation_split_size:float = VALIDATION_SPLIT
                 ):
        self.windows_size = windows_size
        self.train_split_size = train_split_size
        self.validation_split_size = validation_split_size
        self.scaler = StandardScaler()    

    def fit_transform(self, df:pd.DataFrame)-> np.ndarray:
            stock_close = df["Close"]            
            dataset = stock_close.values.reshape(-1, 1)
            
            self.training_end_index = int(np.ceil(len(dataset) * self.train_split_size))
            self.validation_end_index = int(np.ceil(len(dataset) * self.validation_split_size))
            self.y_val_real = stock_close.iloc[self.training_end_index:self.validation_end_index]
            self.y_test_real = stock_close.iloc[self.validation_end_index:]
            self.scaler.fit(dataset[:self.training_end_index])
            scaled_data = self.scaler.transform(dataset)

            return scaled_data
        
    def transform(self, raw_data: np.ndarray, path: str) -> np.ndarray:
        """Scales raw input data using an already-fitted scaler loaded from disk.
        Does not refit. Used at inference time, not during training."""
        loaded_scaler = self.load_scaler(path)
        return loaded_scaler.transform(raw_data)

    # Create sliding windows independently for train, validation, and test splits.
    def create_windows(self, scaled_data:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []

        for i in range(self.windows_size, len(scaled_data)):
            X.append(scaled_data[i - self.windows_size:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)

        X = X.reshape(
            X.shape[0],
            X.shape[1],
            1
        )
        return X, y       

    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)
        print("\n✅ scaler saved in:")
        print(path)
        
    def load_scaler(self, path):
        return joblib.load(path)

    def run_all(self, df:pd.DataFrame) -> dict[str, np.ndarray]:
        scaled_data = self.fit_transform(df)
        training_data = scaled_data[ : self.training_end_index]
        X_train, y_train = self.create_windows(training_data)

        val_data = scaled_data[self.training_end_index - self.windows_size : self.validation_end_index]
        X_val, y_val = self.create_windows(val_data)     

        test_data = scaled_data[self.validation_end_index - self.windows_size : ]
        X_test, y_test = self.create_windows(test_data)     

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val":X_val,
            "y_val":y_val,
            "X_test": X_test,
            "y_test": y_test,
            # "y_test_scaled": y_test_scaled,
            "y_val_real": self.y_val_real,
            "y_test_real": self.y_test_real,
        }                           

if __name__ == "__main__":
    stock_loader = StockLoader()
    
    dataset = stock_loader.fetch()


    preprocess = TimeSeriesPreprocessor()
    data = preprocess.run_all(dataset)
    preprocess.save_scaler('models/scaler.bin')

    
