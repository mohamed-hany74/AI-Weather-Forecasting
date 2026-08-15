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
from ..services.model_service import get_safe_model_info
from .serializers import format_error_response

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def health_check(request: HttpRequest) -> JsonResponse:
    """GET /api/health/ - Service health check."""
    return JsonResponse({
        "status": "ok",
        "service": "WeatherCast AI API",
        "version": "2.0.0"
    }, status=200)


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
def predict_temperature_view(request: HttpRequest) -> JsonResponse:
    """POST /api/predict/ - 24-hour temperature forecast endpoint."""
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
