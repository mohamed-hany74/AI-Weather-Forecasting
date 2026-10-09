"""
Domain Contracts and Constants for WeatherCast AI (Phase 3).
"""

from typing import List

# Frozen Feature Contract (Phase 3: 21 Features)
REQUIRED_FEATURE_COUNT = 21
REQUIRED_HISTORY_LENGTH = 168
FORECAST_HORIZON_HOURS = 24

FEATURE_NAMES: List[str] = [
    "apparent_temperature",
    "pressure_msl",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "cloud_cover",
    "pressure_change_3h",
    "pressure_change_6h",
    "hours_since_daily_max",
    "hour_cos",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "temperature_2m_lag_1",
    "temperature_2m_lag_24",
    "temperature_2m_lag_72",
    "temperature_2m_lag_168",
    "temperature_2m_rolling_max_6",
    "temperature_2m_rolling_max_24",
    "temperature_2m_rolling_mean_24",
    "temperature_2m_rolling_std_24"
]

# Physical bounds for atmospheric plausibility
MIN_ATMOSPHERIC_TEMP = -20.0
MAX_ATMOSPHERIC_TEMP = 60.0
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0

# Exogenous input strict validation bounds
MIN_RELATIVE_HUMIDITY = 0.0
MAX_RELATIVE_HUMIDITY = 100.0
MIN_PRESSURE_MSL = 870.0
MAX_PRESSURE_MSL = 1085.0
MIN_APPARENT_TEMP = -60.0
MAX_APPARENT_TEMP = 60.0
MIN_DEW_POINT = -50.0
MAX_DEW_POINT = 40.0
MIN_WIND_SPEED = 0.0
MAX_WIND_SPEED = 150.0
MIN_CLOUD_COVER = 0.0
MAX_CLOUD_COVER = 100.0

# Empirical Bias Calibration (ERA5 Reanalysis vs Open-Meteo Forecast API)
# Measured in Phase 1 Scientific Truth Audit:
EXOGENOUS_CALIBRATION_OFFSETS = {
    "apparent_temperature": 1.555,
    "relative_humidity_2m": 11.762,
    "pressure_msl": 0.143,
    "dew_point_2m": 0.0,
    "wind_speed_10m": 0.0,
    "cloud_cover": 0.0
}
