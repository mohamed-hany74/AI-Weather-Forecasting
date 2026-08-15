"""
Data structures and schema validation for WeatherCast AI.
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
    MAX_LONGITUDE
)
from .exceptions import ValidationError, TemporalAlignmentError


@dataclass(frozen=True)
class ExogenousWeather:
    apparent_temperature: float
    pressure_msl: float
    relative_humidity_2m: float


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
                    # Treat naive as UTC explicitly
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

        # 4. Optional Exogenous Weather
        exogenous_obj = None
        raw_exo = data.get("exogenous_weather")
        if raw_exo and isinstance(raw_exo, dict):
            try:
                exogenous_obj = ExogenousWeather(
                    apparent_temperature=float(raw_exo["apparent_temperature"]),
                    pressure_msl=float(raw_exo["pressure_msl"]),
                    relative_humidity_2m=float(raw_exo["relative_humidity_2m"])
                )
            except (KeyError, ValueError, TypeError):
                pass

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
    unit: str = "°C"
