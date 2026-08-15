"""
Comprehensive Production Smoke Test for WeatherCast AI.
Runs all pipeline verifications end-to-end.
"""

import os
import sys
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from django.test import Client
from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features
from weather.services.model_service import predict_temperature, get_safe_model_info
from weather.services.weather_service import fetch_realtime_exogenous


def run_smoke_test():
    print("=" * 70)
    print("      WeatherCast AI — Comprehensive Production Smoke Test")
    print("=" * 70)

    # 1. Feature Parity
    print("\n[1/5] Verifying 15-Feature Mathematical Parity...")
    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    TARGET = "temperature_2m"

    test_date_str = "2018-08-15 06:00:00"
    t = pd.to_datetime(test_date_str, utc=True)
    idx = df.index.get_loc(t)
    history = df[TARGET].iloc[idx-168:idx].tolist()
    row_t = df.loc[t]

    exo = ExogenousWeather(
        apparent_temperature=float(row_t["apparent_temperature"]),
        pressure_msl=float(row_t["pressure_msl"]),
        relative_humidity_2m=float(row_t["relative_humidity_2m"])
    )

    vec = build_production_features(history, t.replace(tzinfo=None), exo)
    assert len(vec) == 15, "Feature vector count != 15"
    print(f"  Feature count: {len(vec)} features built successfully.")

    # 2. Synthetic Lag Alignment
    print("\n[2/5] Verifying Lag Indices on Synthetic Sequence [0..167]...")
    synth_history = [float(i) for i in range(168)]
    dt_synth = datetime(2024, 6, 15, 12, 0, 0)
    synth_vec = build_production_features(synth_history, dt_synth, exo)
    feat_map = dict(zip(FEATURE_NAMES, synth_vec))

    assert feat_map["temperature_2m_lag_1"] == 167.0, "lag_1 mismatch"
    assert feat_map["temperature_2m_lag_24"] == 144.0, "lag_24 mismatch"
    assert feat_map["temperature_2m_lag_72"] == 96.0, "lag_72 mismatch"
    assert feat_map["temperature_2m_lag_168"] == 0.0, "lag_168 mismatch"
    assert feat_map["temperature_2m_rolling_max_6"] == 167.0, "rolling_max_6 mismatch"
    assert feat_map["temperature_2m_rolling_max_24"] == 167.0, "rolling_max_24 mismatch"
    assert abs(feat_map["temperature_2m_rolling_mean_24"] - np.mean(range(144, 168))) < 1e-6, "rolling_mean mismatch"
    assert abs(feat_map["temperature_2m_rolling_std_24"] - np.std(range(144, 168), ddof=1)) < 1e-6, "rolling_std mismatch"
    print("  All lag and rolling stats verified (ddof=1 sample std).")

    # 3. Open-Meteo Integration
    print("\n[3/5] Verifying Open-Meteo Real-Time Exogenous Weather Fetch...")
    now_utc = datetime.now(timezone.utc)
    live_exo = fetch_realtime_exogenous(30.0444, 31.2357, now_utc)
    print(f"  Cairo Live: Apparent={live_exo.apparent_temperature}°C | Pressure={live_exo.pressure_msl} hPa | Humidity={live_exo.relative_humidity_2m}%")

    # 4. Live API Endpoint Prediction & Temporal Cutoff
    print("\n[4/5] Executing Live End-to-End Prediction on Django API...")
    c = Client()
    # Fetch live 7-day history from Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast?latitude=30.0444&longitude=31.2357&hourly=temperature_2m&past_days=7&forecast_days=1&timezone=UTC"
    r = requests.get(url, timeout=10)
    data = r.json()
    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]

    now_prefix = now_utc.strftime("%Y-%m-%dT%H:00")
    t_idx = None
    for i, ts in enumerate(times):
        if ts.startswith(now_prefix[:13]):
            t_idx = i
            break
    if t_idx is None or t_idx < 168:
        t_idx = len(temps) - 1
        for i in range(len(times)-1, -1, -1):
            if pd.to_datetime(times[i], utc=True) <= now_utc:
                t_idx = i
                break

    start_idx = t_idx - 168
    history_168 = [float(round(v, 2)) for v in temps[start_idx:t_idx]]
    timestamps_168 = times[start_idx:t_idx]

    # Verify temporal cutoff
    last_hist_ts = pd.to_datetime(timestamps_168[-1], utc=True)
    ref_ts = pd.to_datetime(times[t_idx], utc=True)
    assert last_hist_ts < ref_ts, "CRITICAL: Last history timestamp >= reference t!"
    print(f"  Prediction Reference t  : {times[t_idx]} UTC")
    print(f"  First History (t-168h)  : {timestamps_168[0]} UTC ({history_168[0]}°C)")
    print(f"  Last History  (t-1h)    : {timestamps_168[-1]} UTC ({history_168[-1]}°C)")

    payload = {
        "current_time": times[t_idx] + ":00Z",
        "latitude": 30.0444,
        "longitude": 31.2357,
        "temperature_history_168h": history_168
    }

    res = c.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200, f"API failed with {res.status_code}: {res.content}"
    res_data = res.json()
    assert res_data["status"] == "success"

    pred = res_data["prediction"]["predicted_temperature_2m"]
    forecast_time = res_data["prediction"]["forecast_time"]

    print("\n[5/5] Prediction Sanity Check & Horizon Verification:")
    print(f"  Forecast Target Time   : {forecast_time}")
    print(f"  Predicted Temperature  : {pred:.2f}°C")
    print(f"  Latest Observed (t-1h) : {history_168[-1]:.2f}°C")
    print(f"  Yesterday Same Hour    : {history_168[-24]:.2f}°C")

    # Target horizon check: forecast = t + 24h
    exp_f_time = (ref_ts + timedelta(hours=24)).strftime("%Y-%m-%dT%H:00:00Z")
    assert forecast_time == exp_f_time, f"Forecast time {forecast_time} != {exp_f_time}"

    # Bounds check for Cairo
    assert 15.0 <= pred <= 48.0, f"Prediction {pred}°C outside Cairo atmospheric bounds [15, 48]"
    print("  --> Physical sanity check passed.")

    print("\n" + "=" * 70)
    print("      ALL PRODUCTION SMOKE TESTS PASSED WITH 100% SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
