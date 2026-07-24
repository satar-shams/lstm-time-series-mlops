from fastapi import FastAPI

from app.core.config import settings
from app.api.routes.prediction import router as prediction_router
from app.api.routes.health import router as health_router

from app.core.exceptions import (
    ModelLoadError,
    PredictionFailedError,
)

from app.core.exception_handlers import (
    model_load_exception_handler,
    prediction_failed_exception_handler,
)

app = FastAPI(title=settings.title, version=settings.APP_VERSION)

app.add_exception_handler(
    ModelLoadError,
    model_load_exception_handler,
)

app.add_exception_handler(
    PredictionFailedError,
    prediction_failed_exception_handler,
)

app.include_router(prediction_router)
app.include_router(health_router)

@app.get("/")
def root():
    return {
        "service": settings.title,
        "version": settings.APP_VERSION,
        "status": "running",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)


