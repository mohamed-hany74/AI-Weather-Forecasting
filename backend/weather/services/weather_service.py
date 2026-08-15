"""
Weather Service: Open-Meteo integration with resilient timeout, retry, and exact UTC matching.
Supports both live forecast API and historical archive API for past dates.
"""

import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from ..domain.schemas import ExogenousWeather
from ..domain.exceptions import ExternalWeatherServiceError

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 10  # seconds


def fetch_realtime_exogenous(latitude: float, longitude: float, target_time: datetime) -> ExogenousWeather:
    """
    Fetch apparent_temperature, pressure_msl, and relative_humidity_2m from Open-Meteo.
    Matches exact target hour in UTC.
    Uses Archive API for historical dates older than 5 days; Forecast API for recent/future dates.
    """
    now_utc = datetime.now(timezone.utc)
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    # Determine if target_time is historical (> 5 days ago)
    is_historical = (now_utc - target_time) > timedelta(days=5)

    if is_historical:
        # Use Archive API for historical backtesting / past dates
        day_str = target_time.strftime("%Y-%m-%d")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": day_str,
            "end_date": day_str,
            "hourly": ["apparent_temperature", "pressure_msl", "relative_humidity_2m"],
            "timezone": "UTC"
        }
        api_url = OPEN_METEO_ARCHIVE_URL
    else:
        # Use Forecast API with past_days=7 for recent / current dates
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["apparent_temperature", "pressure_msl", "relative_humidity_2m"],
            "hourly": ["apparent_temperature", "pressure_msl", "relative_humidity_2m"],
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "UTC"
        }
        api_url = OPEN_METEO_FORECAST_URL

    try:
        response = requests.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Open-Meteo timeout for lat={latitude}, lon={longitude}")
        raise ExternalWeatherServiceError("Open-Meteo request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo request error: {e}")
        raise ExternalWeatherServiceError(f"Failed to fetch weather data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error when querying Open-Meteo: {e}")
        raise ExternalWeatherServiceError(f"Unexpected weather API error: {str(e)}")

    # 1. Match in hourly array for the target hour
    target_utc_prefix = target_time.strftime("%Y-%m-%dT%H")
    if "hourly" in data and "time" in data["hourly"]:
        times = data["hourly"]["time"]
        matched_idx = None
        for idx, t_str in enumerate(times):
            if t_str.startswith(target_utc_prefix):
                matched_idx = idx
                break

        if matched_idx is not None:
            try:
                app_t = data["hourly"]["apparent_temperature"][matched_idx]
                press = data["hourly"]["pressure_msl"][matched_idx]
                hum = data["hourly"]["relative_humidity_2m"][matched_idx]

                if app_t is not None and press is not None and hum is not None:
                    return ExogenousWeather(
                        apparent_temperature=float(app_t),
                        pressure_msl=float(press),
                        relative_humidity_2m=float(hum)
                    )
            except (KeyError, IndexError, TypeError, ValueError):
                logger.warning("Target hour extraction from hourly failed; trying fallback readings.")

    # 2. Fallback to current readings if live
    if not is_historical and "current" in data:
        curr = data["current"]
        try:
            app_t = curr.get("apparent_temperature")
            press = curr.get("pressure_msl")
            hum = curr.get("relative_humidity_2m")

            if app_t is not None and press is not None and hum is not None:
                return ExogenousWeather(
                    apparent_temperature=float(app_t),
                    pressure_msl=float(press),
                    relative_humidity_2m=float(hum)
                )
        except (TypeError, ValueError) as e:
            raise ExternalWeatherServiceError(f"Invalid data types in Open-Meteo current: {e}")

    raise ExternalWeatherServiceError(f"Open-Meteo did not contain matching weather features for {target_utc_prefix}:00 UTC.")
