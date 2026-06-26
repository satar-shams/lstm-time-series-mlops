from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("models/best_model.keras")

class PredictRequest(BaseModel):
    data: list[float]

@app.get("/")
def root():
    return {"status": "LSTM API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array(req.data).reshape(1, 30, 1)
    y = model.predict(x)
    return {"prediction": float(y[0][0])}
