"""
Feature Engineering Module for WeatherCast AI.
Strictly implements the 15 frozen production features.
"""

import math
import calendar
import numpy as np
import pandas as pd
from typing import List, Dict, Union

# 15 Frozen Production Features in exact model order
PRODUCTION_FEATURES = [
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
    Compute lag and rolling window features from a 168-hour historical sequence.
    history_168 is ordered [t-168h, t-167h, ..., t-1h].
    history_168[-1] = t-1h
    history_168[-24] = t-24h
    history_168[-72] = t-72h
    history_168[0] = t-168h
    """
    if len(history_168) != 168:
        raise ValueError(f"History must contain exactly 168 observations, got {len(history_168)}")
    
    arr = np.asarray(history_168, dtype=np.float64)
    
    return {
        "temperature_2m_lag_1": float(arr[-1]),
        "temperature_2m_lag_24": float(arr[-24]),
        "temperature_2m_lag_72": float(arr[-72]),
        "temperature_2m_lag_168": float(arr[0]),
        "temperature_2m_rolling_max_6": float(np.max(arr[-6:])),
        "temperature_2m_rolling_max_24": float(np.max(arr[-24:])),
        "temperature_2m_rolling_mean_24": float(np.mean(arr[-24:])),
        "temperature_2m_rolling_std_24": float(np.std(arr[-24:], ddof=1))  # ddof=1 for sample std matching pandas
    }


def engineer_dataframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all 15 features for an hourly DateTimeIndex DataFrame."""
    df = df.copy()
    
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
    
    # Target (t + 24h)
    df["target"] = df[TARGET].shift(-FORECAST_HORIZON)
    
    return df
