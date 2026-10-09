"""
Feature Engineering Module for WeatherCast AI (Phase 3).
Defines and engineers 21 causal features including dynamic spring and exogenous features.
"""

import math
import calendar
import numpy as np
import pandas as pd
from typing import List, Dict, Union

# 21 Production Features in exact model order
PRODUCTION_FEATURES = [
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

TARGET = "temperature_2m"
FORECAST_HORIZON = 24


def compute_cyclical_features(hour: int, month: int, dayofyear: int, is_leap_year: bool) -> Dict[str, float]:
    """Compute exact cyclical calendar features matching the training pipeline."""
    days_in_yr = 366 if is_leap_year else 365

    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    month_cos = float(np.cos(2.0 * np.pi * (month - 1) / 12.0))
    dayofyear_sin = float(np.sin(2.0 * np.pi * (dayofyear - 1) / days_in_yr))
    dayofyear_cos = float(np.cos(2.0 * np.pi * (dayofyear - 1) / days_in_yr))

    return {
        "hour_cos": hour_cos,
        "month_cos": month_cos,
        "dayofyear_sin": dayofyear_sin,
        "dayofyear_cos": dayofyear_cos
    }


def compute_time_series_features(history_168: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Compute lag, rolling window, and hours_since_daily_max features from a 168-hour historical sequence.
    history_168 is ordered [t-168h, t-167h, ..., t-1h].
    history_168[-1] = t-1h
    history_168[-24] = t-24h
    history_168[-72] = t-72h
    history_168[0] = t-168h
    """
    if len(history_168) != 168:
        raise ValueError(f"History must contain exactly 168 observations, got {len(history_168)}")

    arr = np.asarray(history_168, dtype=np.float64)

    # hours_since_daily_max: how many hours ago was the maximum observed in the past 24 hours [t-24h to t-1h]
    last_24 = arr[-24:]
    idx_max = int(np.argmax(last_24))  # 0 (which is t-24h) to 23 (which is t-1h)
    hours_since_daily_max = float(24 - idx_max)

    return {
        "temperature_2m_lag_1": float(arr[-1]),
        "temperature_2m_lag_24": float(arr[-24]),
        "temperature_2m_lag_72": float(arr[-72]),
        "temperature_2m_lag_168": float(arr[0]),
        "temperature_2m_rolling_max_6": float(np.max(arr[-6:])),
        "temperature_2m_rolling_max_24": float(np.max(arr[-24:])),
        "temperature_2m_rolling_mean_24": float(np.mean(arr[-24:])),
        "temperature_2m_rolling_std_24": float(np.std(arr[-24:], ddof=1)),
        "hours_since_daily_max": hours_since_daily_max
    }


def engineer_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all 21 features for an hourly DateTimeIndex DataFrame."""
    df = df.copy()

    # Dynamic pressure change features
    if "pressure_msl" in df.columns:
        df["pressure_change_3h"] = df["pressure_msl"] - df["pressure_msl"].shift(3)
        df["pressure_change_6h"] = df["pressure_msl"] - df["pressure_msl"].shift(6)
    else:
        df["pressure_change_3h"] = 0.0
        df["pressure_change_6h"] = 0.0

    # hours_since_daily_max computed over rolling 24 past hours
    temp_vals = df[TARGET].values
    h_since_max = [np.nan] * 24
    for i in range(24, len(temp_vals)):
        w = temp_vals[i-24:i]
        idx_max = int(np.argmax(w))
        h_since_max.append(float(24 - idx_max))
    df["hours_since_daily_max"] = h_since_max

    # Cyclical
    hour = df.index.hour
    month = df.index.month
    doy = df.index.dayofyear
    days_in_yr = np.where(df.index.is_leap_year, 366, 365)

    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)
    df["dayofyear_sin"] = np.sin(2.0 * np.pi * (doy - 1) / days_in_yr)
    df["dayofyear_cos"] = np.cos(2.0 * np.pi * (doy - 1) / days_in_yr)

    # Lags (strictly past)
    for lag in [1, 24, 72, 168]:
        df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)

    # Rolling stats (shift(1) ensures current t is excluded)
    df[f"{TARGET}_rolling_max_6"] = df[TARGET].shift(1).rolling(6).max()
    df[f"{TARGET}_rolling_max_24"] = df[TARGET].shift(1).rolling(24).max()
    df[f"{TARGET}_rolling_mean_24"] = df[TARGET].shift(1).rolling(24).mean()
    df[f"{TARGET}_rolling_std_24"] = df[TARGET].shift(1).rolling(24).std()

    # Target: Delta target y(t+24) - lag_24
    df["target_abs"] = df[TARGET].shift(-FORECAST_HORIZON)
    df["target"] = df["target_abs"] - df[f"{TARGET}_lag_24"]

    return df
