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

# Empirical Bias Calibration (ERA5 Reanalysis vs Open-Meteo Forecast API)
# Measured in Phase 1 Scientific Truth Audit (2026 live period, 1608 hours):
# Forecast API apparent_temp had -1.555°C bias vs ERA5 -> Add +1.555
# Forecast API relative_humidity had -11.762% bias vs ERA5 -> Add +11.762
# Forecast API pressure_msl had -0.143 hPa bias vs ERA5 -> Add +0.143
EXOGENOUS_CALIBRATION_OFFSETS = {
    "apparent_temperature": 1.555,
    "relative_humidity_2m": 11.762,
    "pressure_msl": 0.143
}

