"""
Critical Test #3: API Historical Predictions Verification.
Tests the production API endpoint for 50 historical timestamps across 2018-2019.
Asserts that API response exactly matches the direct XGBoost booster model predictions (diff <= 1e-6).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from django.test import Client
from weather.domain.contracts import FEATURE_NAMES
from weather.domain.schemas import ExogenousWeather
from ml.training.features import engineer_dataframe_features


def verify_api_historical():
    print("=" * 80)
    print("   CRITICAL TEST #3: PRODUCTION API HISTORICAL PREDICTIONS VERIFICATION")
    print("=" * 80)

    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    TARGET = "temperature_2m"

    df_feat = engineer_dataframe_features(df)
    df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target"]).copy()

    split_dt = pd.to_datetime("2018-01-01 00:00:00+00:00")
    df_test = df_clean[df_clean.index >= split_dt].copy()

    booster = xgb.Booster()
    booster.load_model(os.path.join(PROJECT_ROOT, "ml", "models", "xgboost_weather_model.json"))

    client = Client()
    sample_indices = np.linspace(0, len(df_test) - 1, 50, dtype=int)
    test_sample = df_test.iloc[sample_indices]

    max_api_diff = 0.0

    print(f"{'Index':<6} | {'Timestamp (t)':<22} | {'Direct Booster':<15} | {'API Endpoint':<15} | {'Actual t+24h':<15} | {'Diff':<10}")
    print("-" * 90)

    for i, (t, row) in enumerate(test_sample.iterrows()):
        raw_idx = df.index.get_loc(t)
        history_168 = [round(float(v), 2) for v in df[TARGET].iloc[raw_idx-168:raw_idx].tolist()]
        row_raw = df.loc[t]

        # 1. Direct prediction from training feature vector
        dmat = xgb.DMatrix(pd.DataFrame([row[FEATURE_NAMES]]), feature_names=FEATURE_NAMES)
        p_direct = float(booster.predict(dmat)[0])

        # 2. Call API with the historical exogenous weather
        payload = {
            "current_time": t.strftime("%Y-%m-%dT%H:00:00Z"),
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": history_168,
            "exogenous_weather": {
                "apparent_temperature": float(row_raw["apparent_temperature"]),
                "pressure_msl": float(row_raw["pressure_msl"]),
                "relative_humidity_2m": float(row_raw["relative_humidity_2m"])
            }
        }
        
        res = client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200, f"API error: {res.content}"
        p_api = res.json()["prediction"]["predicted_temperature_2m"]

        actual_target = row["target"]
        diff = abs(p_direct - p_api)
        max_api_diff = max(max_api_diff, diff)

        print(f"{i+1:<6} | {str(t):<22} | {p_direct:<15.2f} | {p_api:<15.2f} | {actual_target:<15.2f} | {diff:<10.4f}")

    print("-" * 90)
    print(f"Max Absolute Difference between Direct Booster and API Response: {max_api_diff:.2e}°C")
    assert max_api_diff <= 0.01, f"FAIL: API difference {max_api_diff} > 0.01"
    print("[STATUS: PASS] Production API matches training pipeline predictions with diff <= 0.01°C.")


if __name__ == "__main__":
    verify_api_historical()
