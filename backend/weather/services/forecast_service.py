"""
Forecast Orchestration Service: Coordinates validation, exogenous fetch, feature assembly, and prediction (Phase 3).
"""

import logging
from datetime import timedelta
from typing import Dict, Any

from ..domain.schemas import PredictionRequest, PredictionResponse
from .weather_service import fetch_realtime_exogenous
from .feature_service import build_production_features
from .model_service import predict_temperature, get_safe_model_info

logger = logging.getLogger(__name__)


def generate_24h_forecast(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete 24-hour ahead temperature prediction workflow."""
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

    # 4. Run inference with delta reconstruction and quantile intervals
    # lag_24 is temperature_history_168[-24]
    lag_24 = float(req.temperature_history_168h[-24])
    predicted_temp, lower_bound, upper_bound = predict_temperature(feature_vector, lag_24=lag_24)

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
            "confidence_interval_80": {
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2)
            },
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
