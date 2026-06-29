from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from app.core.logger import logger
from src.config import APP_HOST, APP_PORT, TICKER, MODEL_PATH, SCALER_PATH

from src.inference.predictor import Predictor
predictor = Predictor(model_path= MODEL_PATH, scaler_path= SCALER_PATH ) 

app = FastAPI(title=f"LSTM {TICKER} Forecaster", version="1.0")

class PredictRequest(BaseModel):
    data: list[float]

@app.get("/")
def root():
    return {f"status": f"LSTM API is running for: {TICKER}"}

@app.post("/predict")
def predict(req: PredictRequest):
    input_data = req.data
    logger.info("Received /predict request with %d values", len(input_data))
    try:
        predict_next_data = predictor.predict(raw_window=input_data)
        logger.info("Prediction successful")
        return {"prediction": predict_next_data}
    except ValueError as e:
        logger.warning("Bad request - invalid input: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal prediction error")


# Start Uvicorn using config
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=True)


