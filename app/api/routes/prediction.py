from fastapi import APIRouter

from app.core.logger import logger
from app.schemas.prediction import PredictRequest, PredictResponse
from src.inference.predictor import Predictor

from app.core.exceptions import PredictionFailedError

router = APIRouter(
    prefix="/api/v1",
    tags=["prediction"],
)

predictor = Predictor()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Received /predict request with %d values", len(req.data))
    try:
        prediction = predictor.predict(raw_window=req.data)
    except ValueError:
        raise  # let FastAPI's built-in validation handling deal with this — see below
    except Exception as e:
        raise PredictionFailedError(f"Prediction failed: {e}") from e
    logger.info("Prediction successful")
    return {"prediction": prediction}