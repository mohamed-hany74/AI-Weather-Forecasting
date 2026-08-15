"""
Domain Exceptions for WeatherCast AI.
"""

from typing import Dict, Any, Optional


class WeatherCastBaseException(Exception):
    """Base exception for all domain errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(WeatherCastBaseException):
    """Raised when input payload fails validation."""
    def __init__(self, message: str, errors: Optional[Dict[str, str]] = None):
        super().__init__(message=message, code="VALIDATION_ERROR", details={"errors": errors or {}})


class TemporalAlignmentError(WeatherCastBaseException):
    """Raised when temporal boundary or sequence is violated (e.g. future data)."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="TEMPORAL_ALIGNMENT_ERROR", details=details)


class ExternalWeatherServiceError(WeatherCastBaseException):
    """Raised when Open-Meteo or external weather provider fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="EXTERNAL_SERVICE_ERROR", details=details)


class ModelInferenceError(WeatherCastBaseException):
    """Raised when XGBoost inference fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, code="MODEL_INFERENCE_ERROR", details=details)
