import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from src.data.loader import StockLoader

class TimeSeriesPreprocessor:
    def __init__(self,
                 windows_size:int = 30,
                 split_size:float = 0.9
                 ):
        self.windows_size = windows_size
        self.split_size = split_size
        self.scaler = StandardScaler()    

    def fit_transform(self, df:pd.DataFrame)-> np.ndarray:
            stock_close = df["Close"]            
            dataset = stock_close.values.reshape(-1, 1)
            
            self.training_data_len  = int(np.ceil(len(dataset) * self.split_size))
            self.y_test_real = stock_close.iloc[self.training_data_len:]
            self.scaler.fit(dataset[:self.training_data_len])
            scaled_data = self.scaler.transform(dataset)

            return scaled_data
        
    def transform(self, raw_data: np.ndarray, path: str) -> np.ndarray:
        """Scales raw input data using an already-fitted scaler loaded from disk.
        Does not refit. Used at inference time, not during training."""
        loaded_scaler = self.load_scaler(path)
        return loaded_scaler.transform(raw_data)

    # in order to use this function, we need to split data for test and train and then call below function for both test and train
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

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path):
        return joblib.load(path)

    def run_all(self, df:pd.DataFrame) -> dict[str, np.ndarray]:
        scaled_data = self.fit_transform(df)
        training_data = scaled_data[:self.training_data_len]
        X_train, y_train = self.create_windows(training_data)

        test_data = scaled_data[self.training_data_len - self.windows_size:]
        X_test, y_test = self.create_windows(test_data)     

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            # "y_test_scaled": y_test_scaled,
            "y_test_real": self.y_test_real,
        }                           

if __name__ == "__main__":
    stock_loader = StockLoader(ticker = "AAPL", 
                               start_date= "2010-01-01",
                               end_date="2026-06-26", auto_adjust= True)
    
    dataset = stock_loader.fetch()


    preprocess = TimeSeriesPreprocessor()
    data = preprocess.run_all(dataset)
    preprocess.save_scaler('models/scaler.bin')
        
    print("y_test shape:", data["y_test"].shape)
    print("y_test_real shape:", data["y_test_real"].shape)
