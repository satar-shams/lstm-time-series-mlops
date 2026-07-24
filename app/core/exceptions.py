# app/core/exceptions.py

class AppException(Exception):
    """Base exception for application-specific errors."""


class PredictionError(AppException):
    """Base exception for prediction-related errors."""


class ModelLoadError(PredictionError):
    """Raised when the ML model or required artifacts cannot be loaded."""


class PredictionFailedError(PredictionError):
    """Raised when prediction fails."""