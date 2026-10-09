"""
Weather Service: Open-Meteo integration with resilient timeout, retry, and exact UTC matching.
Supports both live forecast API and historical archive API for past dates.
"""

import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from django.core.cache import cache
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from ..domain.schemas import ExogenousWeather
from ..domain.exceptions import ExternalWeatherServiceError
from ..domain.contracts import EXOGENOUS_CALIBRATION_OFFSETS

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 5  # Strict 5 seconds timeout
CACHE_TTL = 900      # 15 minutes cache TTL

# Resilient HTTP Session with 1 retry on 500/502/503/504
_session = requests.Session()
_retries = Retry(total=1, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retries)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def get_cache_key(latitude: float, longitude: float, target_time: datetime) -> str:
    """Generate normalized cache key: rounded to 2 decimals + exact UTC hour."""
    norm_lat = round(latitude, 2)
    norm_lon = round(longitude, 2)
    hour_str = target_time.strftime("%Y%m%d%H")
    return f"openmeteo_exo_{norm_lat}_{norm_lon}_{hour_str}"


def calibrate_exogenous(apparent_t: float, press_msl: float, rel_hum: float) -> ExogenousWeather:
    """
    Applies Scenario A Exogenous Bias Calibration (Forecast API -> ERA5 reanalysis).
    Compensates for systematic NWP physical forecast bias against ERA5 training distributions:
      - apparent_temperature: +1.555°C
      - relative_humidity_2m: +11.762% (clipped to [0.0, 100.0])
      - pressure_msl: +0.143 hPa
    """
    calibrated_app_t = apparent_t + EXOGENOUS_CALIBRATION_OFFSETS["apparent_temperature"]
    calibrated_press = press_msl + EXOGENOUS_CALIBRATION_OFFSETS["pressure_msl"]
    raw_hum = rel_hum + EXOGENOUS_CALIBRATION_OFFSETS["relative_humidity_2m"]
    calibrated_hum = max(0.0, min(100.0, raw_hum))

    return ExogenousWeather(
        apparent_temperature=round(calibrated_app_t, 4),
        pressure_msl=round(calibrated_press, 4),
        relative_humidity_2m=round(calibrated_hum, 4)
    )


def fetch_realtime_exogenous(
    latitude: float,
    longitude: float,
    target_time: datetime,
    calibrate: bool = True
) -> ExogenousWeather:
    """
    Fetch apparent_temperature, pressure_msl, and relative_humidity_2m from Open-Meteo.
    Matches exact target hour in UTC.
    Uses 15-minute caching based on rounded coordinates.
    Employs strict 5-second timeout with 1 retry.
    """
    now_utc = datetime.now(timezone.utc)
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    # Check cache first
    cache_key = get_cache_key(latitude, longitude, target_time)
    cached_val = cache.get(cache_key)
    if cached_val is not None and isinstance(cached_val, dict):
        try:
            return ExogenousWeather(
                apparent_temperature=cached_val["apparent_temperature"],
                pressure_msl=cached_val["pressure_msl"],
                relative_humidity_2m=cached_val["relative_humidity_2m"]
            )
        except (KeyError, TypeError):
            pass

    # Determine if target_time is historical (> 5 days ago)
    is_historical = (now_utc - target_time) > timedelta(days=5)

    if is_historical:
        # Use Archive API for historical backtesting / past dates (ERA5 native)
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
        # Strictly queries only the 3 exogenous features trained by the model
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
        response = _session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
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

    def _save_and_return(result: ExogenousWeather) -> ExogenousWeather:
        cache.set(cache_key, {
            "apparent_temperature": result.apparent_temperature,
            "pressure_msl": result.pressure_msl,
            "relative_humidity_2m": result.relative_humidity_2m
        }, CACHE_TTL)
        return result


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
                    raw_app_t = float(app_t)
                    raw_press = float(press)
                    raw_hum = float(hum)

                    # Only calibrate Forecast API data; Archive API is already ERA5
                    if calibrate and not is_historical:
                        res = calibrate_exogenous(raw_app_t, raw_press, raw_hum)
                    else:
                        res = ExogenousWeather(
                            apparent_temperature=raw_app_t,
                            pressure_msl=raw_press,
                            relative_humidity_2m=raw_hum
                        )
                    return _save_and_return(res)
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
                raw_app_t = float(app_t)
                raw_press = float(press)
                raw_hum = float(hum)

                if calibrate:
                    res = calibrate_exogenous(raw_app_t, raw_press, raw_hum)
                else:
                    res = ExogenousWeather(
                        apparent_temperature=raw_app_t,
                        pressure_msl=raw_press,
                        relative_humidity_2m=raw_hum
                    )
                return _save_and_return(res)
        except (TypeError, ValueError) as e:
            raise ExternalWeatherServiceError(f"Invalid data types in Open-Meteo current: {e}")


    raise ExternalWeatherServiceError(f"Open-Meteo did not contain matching weather features for {target_utc_prefix}:00 UTC.")

