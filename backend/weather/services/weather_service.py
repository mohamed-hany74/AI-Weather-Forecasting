"""
Weather Service: Open-Meteo integration with resilient timeout, retry, and exact UTC matching (Phase 3).
Supports both live forecast API and historical archive API for past dates.
Fetches exogenous features: apparent_temp, pressure_msl, relative_humidity, dew_point_2m, wind_speed_10m, cloud_cover.
Computes pressure_change_3h and pressure_change_6h dynamically from the hourly sequence.
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

HOURLY_VARS = [
    "apparent_temperature",
    "pressure_msl",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "cloud_cover"
]

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
    return f"openmeteo_exo_p3_{norm_lat}_{norm_lon}_{hour_str}"


def calibrate_exogenous(
    apparent_t: float,
    press_msl: float,
    rel_hum: float,
    dew: float,
    wind: float,
    cloud: float,
    p_ch3: float,
    p_ch6: float
) -> ExogenousWeather:
    """
    Applies Scenario A Exogenous Bias Calibration (Forecast API -> ERA5 reanalysis).
    """
    calibrated_app_t = apparent_t + EXOGENOUS_CALIBRATION_OFFSETS.get("apparent_temperature", 0.0)
    calibrated_press = press_msl + EXOGENOUS_CALIBRATION_OFFSETS.get("pressure_msl", 0.0)
    raw_hum = rel_hum + EXOGENOUS_CALIBRATION_OFFSETS.get("relative_humidity_2m", 0.0)
    calibrated_hum = max(0.0, min(100.0, raw_hum))

    return ExogenousWeather(
        apparent_temperature=round(calibrated_app_t, 4),
        pressure_msl=round(calibrated_press, 4),
        relative_humidity_2m=round(calibrated_hum, 4),
        dew_point_2m=round(dew, 4),
        wind_speed_10m=round(wind, 4),
        cloud_cover=round(cloud, 4),
        pressure_change_3h=round(p_ch3, 4),
        pressure_change_6h=round(p_ch6, 4)
    )


def fetch_realtime_exogenous(
    latitude: float,
    longitude: float,
    target_time: datetime,
    calibrate: bool = True
) -> ExogenousWeather:
    """
    Fetch exogenous features from Open-Meteo.
    Matches exact target hour in UTC.
    Uses 15-minute caching based on rounded coordinates.
    Computes pressure changes (3h, 6h) from hourly sequence.
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
                relative_humidity_2m=cached_val["relative_humidity_2m"],
                dew_point_2m=cached_val.get("dew_point_2m", 15.0),
                wind_speed_10m=cached_val.get("wind_speed_10m", 10.0),
                cloud_cover=cached_val.get("cloud_cover", 20.0),
                pressure_change_3h=cached_val.get("pressure_change_3h", 0.0),
                pressure_change_6h=cached_val.get("pressure_change_6h", 0.0)
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
            "hourly": HOURLY_VARS,
            "timezone": "UTC"
        }
        api_url = OPEN_METEO_ARCHIVE_URL
    else:
        # Use Forecast API with past_days=2 for pressure change sequence
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": HOURLY_VARS,
            "hourly": HOURLY_VARS,
            "past_days": 2,
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
            "relative_humidity_2m": result.relative_humidity_2m,
            "dew_point_2m": result.dew_point_2m,
            "wind_speed_10m": result.wind_speed_10m,
            "cloud_cover": result.cloud_cover,
            "pressure_change_3h": result.pressure_change_3h,
            "pressure_change_6h": result.pressure_change_6h
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
                h_data = data["hourly"]
                app_t = float(h_data["apparent_temperature"][matched_idx])
                press = float(h_data["pressure_msl"][matched_idx])
                hum = float(h_data["relative_humidity_2m"][matched_idx])
                dew = float(h_data.get("dew_point_2m", [15.0])[matched_idx])
                wind = float(h_data.get("wind_speed_10m", [10.0])[matched_idx])
                cloud = float(h_data.get("cloud_cover", [20.0])[matched_idx])

                # Calculate pressure changes if past hours exist in array
                pressures = h_data["pressure_msl"]
                p_ch3 = press - float(pressures[matched_idx - 3]) if matched_idx >= 3 and pressures[matched_idx - 3] is not None else 0.0
                p_ch6 = press - float(pressures[matched_idx - 6]) if matched_idx >= 6 and pressures[matched_idx - 6] is not None else 0.0

                if calibrate and not is_historical:
                    res = calibrate_exogenous(app_t, press, hum, dew, wind, cloud, p_ch3, p_ch6)
                else:
                    res = ExogenousWeather(
                        apparent_temperature=app_t,
                        pressure_msl=press,
                        relative_humidity_2m=hum,
                        dew_point_2m=dew,
                        wind_speed_10m=wind,
                        cloud_cover=cloud,
                        pressure_change_3h=p_ch3,
                        pressure_change_6h=p_ch6
                    )
                return _save_and_return(res)
            except (KeyError, IndexError, TypeError, ValueError):
                logger.warning("Target hour extraction from hourly failed; trying fallback readings.")

    # 2. Fallback to current readings if live
    if not is_historical and "current" in data:
        curr = data["current"]
        try:
            app_t = float(curr["apparent_temperature"])
            press = float(curr["pressure_msl"])
            hum = float(curr["relative_humidity_2m"])
            dew = float(curr.get("dew_point_2m", 15.0))
            wind = float(curr.get("wind_speed_10m", 10.0))
            cloud = float(curr.get("cloud_cover", 20.0))
            p_ch3 = 0.0
            p_ch6 = 0.0

            if calibrate:
                res = calibrate_exogenous(app_t, press, hum, dew, wind, cloud, p_ch3, p_ch6)
            else:
                res = ExogenousWeather(
                    apparent_temperature=app_t,
                    pressure_msl=press,
                    relative_humidity_2m=hum,
                    dew_point_2m=dew,
                    wind_speed_10m=wind,
                    cloud_cover=cloud,
                    pressure_change_3h=p_ch3,
                    pressure_change_6h=p_ch6
                )
            return _save_and_return(res)
        except (TypeError, ValueError, KeyError) as e:
            raise ExternalWeatherServiceError(f"Invalid data types in Open-Meteo current: {e}")

    raise ExternalWeatherServiceError(f"Open-Meteo did not contain matching weather features for {target_utc_prefix}:00 UTC.")
