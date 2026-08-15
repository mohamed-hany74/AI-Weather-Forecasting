"""
Domain Contracts and Constants for WeatherCast AI.
"""

from typing import List

# Frozen Feature Contract
REQUIRED_FEATURE_COUNT = 15
REQUIRED_HISTORY_LENGTH = 168
FORECAST_HORIZON_HOURS = 24

FEATURE_NAMES: List[str] = [
    "apparent_temperature",
    "pressure_msl",
    "relative_humidity_2m",
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

# Physical bounds for Cairo and atmospheric plausibility
MIN_ATMOSPHERIC_TEMP = -20.0
MAX_ATMOSPHERIC_TEMP = 60.0
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
