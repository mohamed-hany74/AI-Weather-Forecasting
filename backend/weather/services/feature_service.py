"""
Feature Service: Builds and validates the 15 production features.
"""

import math
import calendar
import numpy as np
from datetime import datetime
from typing import List, Dict

from ..domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from ..domain.schemas import ExogenousWeather
from ..domain.exceptions import ValidationError


def build_production_features(
    temperature_history_168: List[float],
    reference_time: datetime,
    exogenous: ExogenousWeather
) -> List[float]:
    """
    Build the exact 15 frozen features in the exact contract order.
    history_168 is ordered [t-168h, ..., t-1h].
    """
    if len(temperature_history_168) != 168:
        raise ValidationError(f"History must contain exactly 168 values, got {len(temperature_history_168)}")

    arr = np.asarray(temperature_history_168, dtype=np.float64)

    # 1. Cyclical time features
    hour = reference_time.hour
    month = reference_time.month
    doy = reference_time.timetuple().tm_yday
    days_in_yr = 366 if calendar.isleap(reference_time.year) else 365

    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    month_cos = float(np.cos(2.0 * np.pi * (month - 1) / 12.0))
    dayofyear_sin = float(np.sin(2.0 * np.pi * (doy - 1) / days_in_yr))
    dayofyear_cos = float(np.cos(2.0 * np.pi * (doy - 1) / days_in_yr))

    # 2. Lags (history[-1] = t-1h)
    lag_1 = float(arr[-1])
    lag_24 = float(arr[-24])
    lag_72 = float(arr[-72])
    lag_168 = float(arr[0])

    # 3. Rolling statistics
    rolling_max_6 = float(np.max(arr[-6:]))
    rolling_max_24 = float(np.max(arr[-24:]))
    rolling_mean_24 = float(np.mean(arr[-24:]))
    rolling_std_24 = float(np.std(arr[-24:], ddof=1))  # sample std matching training

    feature_dict = {
        "apparent_temperature": float(exogenous.apparent_temperature),
        "pressure_msl": float(exogenous.pressure_msl),
        "relative_humidity_2m": float(exogenous.relative_humidity_2m),
        "hour_cos": hour_cos,
        "month_cos": month_cos,
        "dayofyear_sin": dayofyear_sin,
        "dayofyear_cos": dayofyear_cos,
        "temperature_2m_lag_1": lag_1,
        "temperature_2m_lag_24": lag_24,
        "temperature_2m_lag_72": lag_72,
        "temperature_2m_lag_168": lag_168,
        "temperature_2m_rolling_max_6": rolling_max_6,
        "temperature_2m_rolling_max_24": rolling_max_24,
        "temperature_2m_rolling_mean_24": rolling_mean_24,
        "temperature_2m_rolling_std_24": rolling_std_24,
    }

    # Strict ordering
    vector = [feature_dict[name] for name in FEATURE_NAMES]
    assert len(vector) == REQUIRED_FEATURE_COUNT, f"Vector length {len(vector)} != {REQUIRED_FEATURE_COUNT}"
    return vector
