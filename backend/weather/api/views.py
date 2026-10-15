"""
API Views for WeatherCast AI.
"""

import json
import logging
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from ..domain.exceptions import (
    ValidationError,
    TemporalAlignmentError,
    ExternalWeatherServiceError,
    ModelInferenceError,
    WeatherCastBaseException
)
from ..services.forecast_service import generate_24h_forecast
from ..services.model_service import get_safe_model_info, is_model_ready
from .serializers import format_error_response
from .throttling import ratelimit

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def health_check(request: HttpRequest) -> JsonResponse:
    """GET /api/health/ - Service liveness probe."""
    return JsonResponse({
        "status": "ok",
        "service": "WeatherCast AI API",
        "version": "2.0.0"
    }, status=200)


@require_http_methods(["GET"])
def readiness_check(request: HttpRequest) -> JsonResponse:
    """
    GET /api/ready/ - Kubernetes/Container readiness probe.
    Returns 200 {"status": "ready"} if model is initialized with 15 features;
    returns 503 Service Unavailable if uninitialized or misconfigured.
    """
    if is_model_ready():
        return JsonResponse({"status": "ready"}, status=200)
    return JsonResponse(
        format_error_response("Model is not initialized or expects invalid feature contract.", code="NOT_READY"),
        status=503
    )


@require_http_methods(["GET"])
def model_info(request: HttpRequest) -> JsonResponse:
    """GET /api/model-info/ - Model metadata and configuration."""
    try:
        info = get_safe_model_info()
        return JsonResponse({
            "status": "success",
            "model": info
        }, status=200)
    except Exception as e:
        logger.error(f"Error fetching model info: {e}", exc_info=True)
        return JsonResponse(
            format_error_response("Failed to retrieve model info.", code="MODEL_ERROR"),
            status=500
        )


@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(requests_per_minute=60)
def predict_temperature_view(request: HttpRequest) -> JsonResponse:
    """POST /api/predict/ - 24-hour temperature forecast endpoint throttled to 60 req/min."""

    # 1. Parse JSON body
    try:
        if not request.body:
            return JsonResponse(
                format_error_response("Empty request body. JSON payload required.", code="EMPTY_BODY"),
                status=400
            )
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JsonResponse(
            format_error_response("Malformed JSON payload.", code="JSON_PARSE_ERROR", details=str(e)),
            status=400
        )

    # 2. Run forecast pipeline
    try:
        result = generate_24h_forecast(data)
        return JsonResponse(result, status=200)
    except ValidationError as e:
        return JsonResponse(
            format_error_response(e.message, code=e.code, details=e.details),
            status=400
        )
    except TemporalAlignmentError as e:
        return JsonResponse(
            format_error_response(e.message, code=e.code, details=e.details),
            status=400
        )
    except ExternalWeatherServiceError as e:
        logger.warning(f"External service failure: {e.message}")
        return JsonResponse(
            format_error_response(e.message, code=e.code, details=e.details),
            status=503
        )
    except ModelInferenceError as e:
        logger.error(f"Inference failure: {e.message}")
        return JsonResponse(
            format_error_response(e.message, code=e.code),
            status=500
        )
    except Exception as e:
        logger.exception("Unexpected unhandled API error")
        return JsonResponse(
            format_error_response("An unexpected internal error occurred.", code="INTERNAL_SERVER_ERROR"),
            status=500
        )


@csrf_exempt
@require_http_methods(["POST"])
@ratelimit(requests_per_minute=60)
def forecast_easy_view(request: HttpRequest) -> JsonResponse:
    """
    POST /api/forecast/ - Easy-input 24-hour temperature forecast endpoint.
    Accepts minimal parameters:
      {"latitude": 30.0444, "longitude": 31.2357, "current_time": "..."}
    All fields are optional (defaults to Cairo, Egypt and current UTC time).
    Fetches real verified 168h observations from Open-Meteo and executes production XGBoost inference.
    """
    data = {}
    if request.body:
        try:
            data = json.loads(request.body.decode("utf-8"))
            if not isinstance(data, dict):
                return JsonResponse(
                    format_error_response("Payload must be a valid JSON object.", code="VALIDATION_ERROR"),
                    status=400
                )
        except json.JSONDecodeError as e:
            return JsonResponse(
                format_error_response("Malformed JSON payload.", code="JSON_PARSE_ERROR", details=str(e)),
                status=400
            )

    lat = data.get("latitude", 30.0444)
    lon = data.get("longitude", 31.2357)
    raw_time = data.get("current_time")

    try:
        lat = float(lat)
        lon = float(lon)
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return JsonResponse(
                format_error_response("Coordinates out of physical bounds.", code="VALIDATION_ERROR"),
                status=400
            )
    except (TypeError, ValueError):
        return JsonResponse(
            format_error_response("Latitude and longitude must be valid floating point numbers.", code="VALIDATION_ERROR"),
            status=400
        )

    ref_time = None
    if raw_time is not None:
        if not isinstance(raw_time, str):
            return JsonResponse(
                format_error_response("current_time must be an ISO-8601 string.", code="VALIDATION_ERROR"),
                status=400
            )
        try:
            clean_str = raw_time.replace("Z", "+00:00")
            from datetime import datetime, timezone
            ref_time = datetime.fromisoformat(clean_str)
            if ref_time.tzinfo is not None:
                ref_time = ref_time.astimezone(timezone.utc)
            else:
                ref_time = ref_time.replace(tzinfo=timezone.utc)
        except ValueError:
            return JsonResponse(
                format_error_response(f"Invalid ISO-8601 timestamp: '{raw_time}'.", code="VALIDATION_ERROR"),
                status=400
            )

    try:
        from ..services.forecast_service import generate_easy_forecast
        result = generate_easy_forecast(latitude=lat, longitude=lon, reference_time=ref_time)
        return JsonResponse(result, status=200)
    except ExternalWeatherServiceError as e:
        logger.warning(f"External weather service error during easy forecast: {e.message}")
        return JsonResponse(
            format_error_response(e.message, code=e.code, details=e.details),
            status=503
        )
    except ModelInferenceError as e:
        logger.error(f"Inference failure during easy forecast: {e.message}")
        return JsonResponse(
            format_error_response(e.message, code=e.code),
            status=500
        )
    except Exception as e:
        logger.exception("Unexpected error in forecast_easy_view")
        return JsonResponse(
            format_error_response("An unexpected internal error occurred.", code="INTERNAL_SERVER_ERROR"),
            status=500
        )
