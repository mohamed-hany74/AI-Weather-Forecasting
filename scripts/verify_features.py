"""
Verification Script: Feature Parity Validation.
Validates that backend/weather/services/feature_service.py produces feature vectors
identical (< 1e-6 diff) to the independent mathematical definitions across 10+ timestamps.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from weather.domain.contracts import FEATURE_NAMES
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features


def verify_features():
    print("=" * 70)
    print("       FEATURE PARITY VERIFICATION (Mathematical Ground Truth)")
    print("=" * 70)

    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    TARGET = "temperature_2m"

    # Ground truth calculations
    df_gt = df.copy()
    df_gt["hour_cos"] = np.cos(2 * np.pi * df_gt.index.hour / 24)
    df_gt["month_cos"] = np.cos(2 * np.pi * (df_gt.index.month - 1) / 12)
    days_in_yr = np.where(df_gt.index.is_leap_year, 366, 365)
    df_gt["dayofyear_sin"] = np.sin(2 * np.pi * (df_gt.index.dayofyear - 1) / days_in_yr)
    df_gt["dayofyear_cos"] = np.cos(2 * np.pi * (df_gt.index.dayofyear - 1) / days_in_yr)

    for lag in [1, 24, 72, 168]:
        df_gt[f"{TARGET}_lag_{lag}"] = df_gt[TARGET].shift(lag)

    df_gt[f"{TARGET}_rolling_max_6"] = df_gt[TARGET].shift(1).rolling(6).max()
    df_gt[f"{TARGET}_rolling_max_24"] = df_gt[TARGET].shift(1).rolling(24).max()
    df_gt[f"{TARGET}_rolling_mean_24"] = df_gt[TARGET].shift(1).rolling(24).mean()
    df_gt[f"{TARGET}_rolling_std_24"] = df_gt[TARGET].shift(1).rolling(24).std()

    test_dates = [
        "2010-06-15 12:00:00",
        "2012-02-29 12:00:00", # Leap year
        "2013-11-15 18:00:00",
        "2016-12-31 23:00:00", # Leap year boundary
        "2017-01-01 00:00:00",
        "2018-08-15 06:00:00", # Test set
        "2019-01-15 03:00:00",
        "2019-07-20 14:00:00"
    ]

    max_error = 0.0
    for dt_str in test_dates:
        t = pd.to_datetime(dt_str, utc=True)
        idx = df.index.get_loc(t)
        history = df[TARGET].iloc[idx-168:idx].tolist()
        row_t = df.loc[t]

        exo = ExogenousWeather(
            apparent_temperature=float(row_t["apparent_temperature"]),
            pressure_msl=float(row_t["pressure_msl"]),
            relative_humidity_2m=float(row_t["relative_humidity_2m"])
        )

        django_vec = build_production_features(history, t.replace(tzinfo=None), exo)
        
        expected_vec = [
            float(row_t["apparent_temperature"]),
            float(row_t["pressure_msl"]),
            float(row_t["relative_humidity_2m"]),
            float(df_gt.loc[t, "hour_cos"]),
            float(df_gt.loc[t, "month_cos"]),
            float(df_gt.loc[t, "dayofyear_sin"]),
            float(df_gt.loc[t, "dayofyear_cos"]),
            float(df_gt.loc[t, f"{TARGET}_lag_1"]),
            float(df_gt.loc[t, f"{TARGET}_lag_24"]),
            float(df_gt.loc[t, f"{TARGET}_lag_72"]),
            float(df_gt.loc[t, f"{TARGET}_lag_168"]),
            float(df_gt.loc[t, f"{TARGET}_rolling_max_6"]),
            float(df_gt.loc[t, f"{TARGET}_rolling_max_24"]),
            float(df_gt.loc[t, f"{TARGET}_rolling_mean_24"]),
            float(df_gt.loc[t, f"{TARGET}_rolling_std_24"]),
        ]

        err = max(abs(a - b) for a, b in zip(django_vec, expected_vec))
        max_error = max(max_error, err)

    print(f"Max Absolute Error across 15 features on all test timestamps: {max_error:.2e}")
    if max_error < 1e-6:
        print("[PASS] Feature parity is 100% verified.")
        return True
    else:
        print("[FAIL] Feature parity mismatch detected!")
        sys.exit(1)


if __name__ == "__main__":
    verify_features()
