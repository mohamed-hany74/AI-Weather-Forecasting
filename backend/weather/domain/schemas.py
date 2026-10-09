"""
Data structures and schema validation for WeatherCast AI (Phase 3).
"""

import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .contracts import (
    REQUIRED_HISTORY_LENGTH,
    MIN_ATMOSPHERIC_TEMP,
    MAX_ATMOSPHERIC_TEMP,
    MIN_LATITUDE,
    MAX_LATITUDE,
    MIN_LONGITUDE,
    MAX_LONGITUDE,
    MIN_RELATIVE_HUMIDITY,
    MAX_RELATIVE_HUMIDITY,
    MIN_PRESSURE_MSL,
    MAX_PRESSURE_MSL,
    MIN_APPARENT_TEMP,
    MAX_APPARENT_TEMP,
    MIN_DEW_POINT,
    MAX_DEW_POINT,
    MIN_WIND_SPEED,
    MAX_WIND_SPEED,
    MIN_CLOUD_COVER,
    MAX_CLOUD_COVER
)
from .exceptions import ValidationError, TemporalAlignmentError


@dataclass(frozen=True)
class ExogenousWeather:
    apparent_temperature: float
    pressure_msl: float
    relative_humidity_2m: float
    dew_point_2m: float = 15.0
    wind_speed_10m: float = 10.0
    cloud_cover: float = 20.0
    pressure_change_3h: float = 0.0
    pressure_change_6h: float = 0.0


@dataclass(frozen=True)
class PredictionRequest:
    reference_time: datetime
    latitude: float
    longitude: float
    temperature_history_168h: List[float]
    exogenous_weather: Optional[ExogenousWeather] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionRequest":
        if not isinstance(data, dict):
            raise ValidationError("Payload must be a valid JSON object.")

        errors: Dict[str, str] = {}

        # 1. Reference Time
        raw_time = data.get("current_time")
        parsed_time: Optional[datetime] = None
        if not raw_time:
            errors["current_time"] = "Field 'current_time' is required."
        elif not isinstance(raw_time, str):
            errors["current_time"] = "Field 'current_time' must be an ISO-8601 string."
        else:
            try:
                clean_time_str = raw_time.replace("Z", "+00:00")
                parsed_time = datetime.fromisoformat(clean_time_str)
                if parsed_time.tzinfo is not None:
                    parsed_time = parsed_time.astimezone(timezone.utc)
                else:
                    parsed_time = parsed_time.replace(tzinfo=timezone.utc)
            except ValueError:
                errors["current_time"] = f"Invalid ISO-8601 timestamp: '{raw_time}'."

        # 2. Coordinates
        raw_lat = data.get("latitude")
        raw_lon = data.get("longitude")
        lat_val: Optional[float] = None
        lon_val: Optional[float] = None

        if raw_lat is None:
            errors["latitude"] = "Field 'latitude' is required."
        else:
            try:
                lat_val = float(raw_lat)
                if not (MIN_LATITUDE <= lat_val <= MAX_LATITUDE) or math.isnan(lat_val) or math.isinf(lat_val):
                    errors["latitude"] = f"Latitude must be a finite float between {MIN_LATITUDE} and {MAX_LATITUDE}."
            except (ValueError, TypeError):
                errors["latitude"] = f"Invalid latitude: '{raw_lat}'."

        if raw_lon is None:
            errors["longitude"] = "Field 'longitude' is required."
        else:
            try:
                lon_val = float(raw_lon)
                if not (MIN_LONGITUDE <= lon_val <= MAX_LONGITUDE) or math.isnan(lon_val) or math.isinf(lon_val):
                    errors["longitude"] = f"Longitude must be a finite float between {MIN_LONGITUDE} and {MAX_LONGITUDE}."
            except (ValueError, TypeError):
                errors["longitude"] = f"Invalid longitude: '{raw_lon}'."

        # 3. 168h History
        raw_history = data.get("temperature_history_168h")
        valid_history: List[float] = []

        if raw_history is None:
            errors["temperature_history_168h"] = "Field 'temperature_history_168h' is required."
        elif not isinstance(raw_history, list):
            errors["temperature_history_168h"] = "Field 'temperature_history_168h' must be an array of floats."
        elif len(raw_history) != REQUIRED_HISTORY_LENGTH:
            errors["temperature_history_168h"] = (
                f"Expected exactly {REQUIRED_HISTORY_LENGTH} readings, got {len(raw_history)}."
            )
        else:
            for idx, val in enumerate(raw_history):
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        errors["temperature_history_168h"] = f"Index {idx} contains NaN or Infinity."
                        break
                    if not (MIN_ATMOSPHERIC_TEMP <= fval <= MAX_ATMOSPHERIC_TEMP):
                        errors["temperature_history_168h"] = (
                            f"Index {idx} value ({fval}°C) outside atmospheric limits [{MIN_ATMOSPHERIC_TEMP}, {MAX_ATMOSPHERIC_TEMP}]."
                        )
                        break
                    valid_history.append(fval)
                except (ValueError, TypeError):
                    errors["temperature_history_168h"] = f"Index {idx} is not a valid number: '{val}'."
                    break

        # 4. Optional Exogenous Weather (Strict validation if provided)
        exogenous_obj = None
        if "exogenous_weather" in data:
            raw_exo = data.get("exogenous_weather")
            if not isinstance(raw_exo, dict):
                errors["exogenous_weather"] = "Field 'exogenous_weather' must be a JSON object."
            else:
                exo_fields = ["apparent_temperature", "pressure_msl", "relative_humidity_2m"]
                missing = [f for f in exo_fields if f not in raw_exo or raw_exo[f] is None]
                if missing:
                    errors["exogenous_weather"] = f"Missing required exogenous keys: {', '.join(missing)}."
                else:
                    exo_vals = {}
                    all_possible_fields = [
                        "apparent_temperature", "pressure_msl", "relative_humidity_2m",
                        "dew_point_2m", "wind_speed_10m", "cloud_cover",
                        "pressure_change_3h", "pressure_change_6h"
                    ]
                    for field in all_possible_fields:
                        if field in raw_exo:
                            try:
                                val = float(raw_exo[field])
                                if not math.isfinite(val):
                                    errors[f"exogenous_weather.{field}"] = f"Field '{field}' must be a finite number."
                                else:
                                    exo_vals[field] = val
                            except (ValueError, TypeError):
                                errors[f"exogenous_weather.{field}"] = f"Field '{field}' must be a valid float."

                    if not errors:
                        hum = float(raw_exo["relative_humidity_2m"])
                        press = float(raw_exo["pressure_msl"])
                        app_t = float(raw_exo["apparent_temperature"])

                        if not (MIN_RELATIVE_HUMIDITY <= hum <= MAX_RELATIVE_HUMIDITY):
                            errors["exogenous_weather.relative_humidity_2m"] = (
                                f"relative_humidity_2m ({hum}%) out of bounds [{MIN_RELATIVE_HUMIDITY}, {MAX_RELATIVE_HUMIDITY}]."
                            )
                        if not (MIN_PRESSURE_MSL <= press <= MAX_PRESSURE_MSL):
                            errors["exogenous_weather.pressure_msl"] = (
                                f"pressure_msl ({press} hPa) out of bounds [{MIN_PRESSURE_MSL}, {MAX_PRESSURE_MSL}]."
                            )
                        if not (MIN_APPARENT_TEMP <= app_t <= MAX_APPARENT_TEMP):
                            errors["exogenous_weather.apparent_temperature"] = (
                                f"apparent_temperature ({app_t}°C) out of bounds [{MIN_APPARENT_TEMP}, {MAX_APPARENT_TEMP}]."
                            )

                        dew = exo_vals.get("dew_point_2m", 15.0)
                        wind = exo_vals.get("wind_speed_10m", 10.0)
                        cloud = exo_vals.get("cloud_cover", 20.0)

                        if "dew_point_2m" in exo_vals and not (MIN_DEW_POINT <= dew <= MAX_DEW_POINT):
                            errors["exogenous_weather.dew_point_2m"] = f"dew_point_2m ({dew}°C) out of bounds."
                        if "wind_speed_10m" in exo_vals and not (MIN_WIND_SPEED <= wind <= MAX_WIND_SPEED):
                            errors["exogenous_weather.wind_speed_10m"] = f"wind_speed_10m ({wind} km/h) out of bounds."
                        if "cloud_cover" in exo_vals and not (MIN_CLOUD_COVER <= cloud <= MAX_CLOUD_COVER):
                            errors["exogenous_weather.cloud_cover"] = f"cloud_cover ({cloud}%) out of bounds."

                        if not errors:
                            exogenous_obj = ExogenousWeather(
                                apparent_temperature=app_t,
                                pressure_msl=press,
                                relative_humidity_2m=hum,
                                dew_point_2m=dew,
                                wind_speed_10m=wind,
                                cloud_cover=cloud,
                                pressure_change_3h=exo_vals.get("pressure_change_3h", 0.0),
                                pressure_change_6h=exo_vals.get("pressure_change_6h", 0.0)
                            )

        if errors:
            raise ValidationError("Payload schema validation failed.", errors=errors)

        return cls(
            reference_time=parsed_time,
            latitude=lat_val,
            longitude=lon_val,
            temperature_history_168h=valid_history,
            exogenous_weather=exogenous_obj
        )


@dataclass(frozen=True)
class PredictionResponse:
    target_variable: str
    forecast_horizon_hours: int
    forecast_time: str
    predicted_temperature_2m: float
    confidence_interval_80: Dict[str, float]
    unit: str = "°C"
