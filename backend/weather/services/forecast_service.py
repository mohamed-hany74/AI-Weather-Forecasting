"""
Forecast Orchestration Service: Coordinates validation, exogenous fetch, feature assembly, and prediction (Phase 3).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from ..domain.schemas import PredictionRequest, PredictionResponse
from ..domain.contracts import TYPICAL_ERROR_CELSIUS, TYPICAL_ERROR_NOTE
from .weather_service import fetch_realtime_exogenous, fetch_168h_history_and_exogenous
from .feature_service import build_production_features
from .model_service import predict_temperature, get_safe_model_info

logger = logging.getLogger(__name__)


def generate_24h_forecast(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute 24-hour ahead temperature prediction workflow from explicit 168h history.
    Time alignment:
      reference_time = t
      req.temperature_history_168h = [t-168h, ..., t-1h]
      history[-1] = temperature at t-1h (the latest historical observation)
      lag_24 = temperature at t-24h (the anchor baseline)
      forecast_dt = t + 24h
      current_temperature = history[-1] (latest observed temperature prior to t)
      change_vs_current = predicted_temperature_2m - current_temperature
    """
    # 1. Parse and validate input schema
    req = PredictionRequest.from_dict(request_data)

    # 2. Fetch exogenous weather at reference time t (or use supplied)
    if req.exogenous_weather is not None:
        exogenous = req.exogenous_weather
    else:
        exogenous = fetch_realtime_exogenous(
            latitude=req.latitude,
            longitude=req.longitude,
            target_time=req.reference_time
        )

    # 3. Assemble 21 production features
    feature_vector = build_production_features(
        temperature_history_168=req.temperature_history_168h,
        reference_time=req.reference_time,
        exogenous=exogenous
    )

    # 4. Run point inference with delta reconstruction: pred_abs = lag_24 + predicted_delta
    lag_24 = float(req.temperature_history_168h[-24])
    current_temp = float(req.temperature_history_168h[-1])
    predicted_temp = predict_temperature(feature_vector, lag_24=lag_24)

    # 5. Compute forecast timestamp (t + 24h)
    forecast_dt = req.reference_time + timedelta(hours=24)
    model_info = get_safe_model_info()

    return {
        "status": "success",
        "prediction": {
            "target_variable": "temperature_2m",
            "forecast_horizon_hours": 24,
            "forecast_time": forecast_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "predicted_temperature_2m": round(predicted_temp, 2),
            "current_temperature": round(current_temp, 2),
            "change_vs_current": round(predicted_temp - current_temp, 2),
            "typical_error_celsius": TYPICAL_ERROR_CELSIUS,
            "error_reference_note": TYPICAL_ERROR_NOTE,
            "unit": "°C"
        },
        "model": {
            "type": model_info["model_type"],
            "target_formulation": model_info.get("target_formulation", "delta_lag24"),
            "feature_count": model_info["feature_count"],
            "metrics": model_info["metrics"]
        },
        "location": {
            "latitude": req.latitude,
            "longitude": req.longitude
        },
        "context": {
            "reference_time": req.reference_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lag_24_anchor": round(lag_24, 2),
            "exogenous_weather_used": {
                "apparent_temperature": exogenous.apparent_temperature,
                "pressure_msl": exogenous.pressure_msl,
                "relative_humidity_2m": exogenous.relative_humidity_2m,
                "dew_point_2m": exogenous.dew_point_2m,
                "wind_speed_10m": exogenous.wind_speed_10m,
                "cloud_cover": exogenous.cloud_cover
            },
            "features_engineered": dict(zip(model_info["features"], feature_vector))
        }
    }


def generate_easy_forecast(
    latitude: float = 30.0444,
    longitude: float = 31.2357,
    reference_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Easy-Input Forecast Workflow:
    Accepts latitude and longitude (defaulting to Cairo: 30.0444, 31.2357) and optional reference time.
    Fetches real verified 168h temperature history and aligned exogenous features from Open-Meteo.
    Builds the exact 21 features with zero synthetic heuristics and executes production inference.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    # 1. Fetch real historical 168h observations and aligned exogenous features from Open-Meteo
    weather_bundle = fetch_168h_history_and_exogenous(
        latitude=latitude,
        longitude=longitude,
        reference_time=reference_time
    )

    history_168 = weather_bundle["history_168"]
    current_temp = weather_bundle["current_temperature"]
    exogenous = weather_bundle["exogenous"]
    ref_hour = weather_bundle["reference_time"]

    # 2. Build 21 features
    feature_vector = build_production_features(
        temperature_history_168=history_168,
        reference_time=ref_hour,
        exogenous=exogenous
    )

    # 3. Predict
    lag_24 = float(history_168[-24])
    predicted_temp = predict_temperature(feature_vector, lag_24=lag_24)

    # 4. Build response
    forecast_dt = ref_hour + timedelta(hours=24)
    model_info = get_safe_model_info()

    return {
        "status": "success",
        "prediction": {
            "target_variable": "temperature_2m",
            "forecast_horizon_hours": 24,
            "forecast_time": forecast_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "predicted_temperature_2m": round(predicted_temp, 2),
            "current_temperature": round(current_temp, 2),
            "change_vs_current": round(predicted_temp - current_temp, 2),
            "typical_error_celsius": TYPICAL_ERROR_CELSIUS,
            "error_reference_note": TYPICAL_ERROR_NOTE,
            "unit": "°C"
        },
        "model": {
            "type": model_info["model_type"],
            "target_formulation": model_info.get("target_formulation", "delta_lag24"),
            "feature_count": model_info["feature_count"],
            "metrics": model_info["metrics"]
        },
        "location": {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4)
        },
        "context": {
            "reference_time": ref_hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lag_24_anchor": round(lag_24, 2),
            "exogenous_weather_used": {
                "apparent_temperature": exogenous.apparent_temperature,
                "pressure_msl": exogenous.pressure_msl,
                "relative_humidity_2m": exogenous.relative_humidity_2m,
                "dew_point_2m": exogenous.dew_point_2m,
                "wind_speed_10m": exogenous.wind_speed_10m,
                "cloud_cover": exogenous.cloud_cover
            },
            "features_engineered": dict(zip(model_info["features"], feature_vector))
        }
    }
