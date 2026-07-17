import numpy as np
from src.models.lstm_model import LSTMForecaster
from tensorflow.keras.callbacks import Callback

class SingleModelTrainer:
    def __init__(self):
        self.forecaster = LSTMForecaster()

    def train(
            self, 
            cfg:dict[str, float | int], 
            X:np.ndarray, 
            y:np.ndarray,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            callbacks: list[Callback] | None = None
            ):
                
        model = self.forecaster.build(input_shape=(X.shape[1], 1),
                                lstm_units= cfg["lstm_units"],
                                dense_units= cfg["dense_units"],
                                dropout_rate= cfg["dropout_rate"],
                               )
        
        self.forecaster.compile_model(model,
                                    cfg["learning_rate"],
                                    cfg["clip_norm"]
                                    )
        
        history = model.fit(
            X, 
            y,
            validation_data=validation_data,
            epochs=cfg["epochs"], 
            batch_size=cfg["batch_size"], 
            callbacks=callbacks,
            verbose=0,
        )

        return model, history





