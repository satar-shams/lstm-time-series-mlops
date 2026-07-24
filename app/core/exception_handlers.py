from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ModelLoadError,
    PredictionFailedError,
)
from app.core.logger import logger


async def model_load_exception_handler(
    request: Request,
    exc: ModelLoadError,
):
    logger.error(
        "Model loading failed: %s",
        str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "The prediction service is currently unavailable.",
        },
    )


async def prediction_failed_exception_handler(
    request: Request,
    exc: PredictionFailedError,
):
    logger.error(
        "Prediction failed: %s",
        str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction could not be completed.",
        },
    )