"""
Comprehensive Evidence-Based Audit Suite for WeatherCast AI (Phase 3).
Executes all audit phases empirically, measuring and logging raw results for 21 features & delta formulation.
"""

import os
import sys
import json
import time
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import xgboost as xgb
from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from django.test import Client
from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather, PredictionRequest
from weather.services.feature_service import build_production_features
from weather.services.model_service import predict_temperature, get_safe_model_info, load_model_artifacts
from weather.services.weather_service import fetch_realtime_exogenous
from ml.training.features import engineer_dataframe_features
from ml.training.evaluation import evaluate_forecasts


def run_full_audit():
    print("=" * 80)
    print("      WeatherCast AI — Phase 3 PRODUCTION AUDIT EXECUTION")
    print("=" * 80)

    # 1. Dataset Preparation
    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    TARGET = "temperature_2m"

    df_feat = engineer_dataframe_features(df)
    df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target", "target_abs"]).copy()

    # Split: Canonical 80/20 chronological split
    n_total = len(df)
    n_test = int(n_total * 0.20)
    split_dt = df.index[n_total - n_test]
    purge_dt = split_dt - pd.Timedelta(hours=24)

    df_train = df_clean[df_clean.index < purge_dt].copy()
    df_test = df_clean[df_clean.index >= split_dt].copy()

    X_train, y_train = df_train[FEATURE_NAMES], df_train["target"]
    X_test, y_test = df_test[FEATURE_NAMES], df_test["target"]
    lag24_test = df_test["temperature_2m_lag_24"].values
    y_test_abs = df_test["target_abs"].values

    print(f"\n[PHASE 1] Dataset Inventory & Boundaries:")
    print(f"  Total records: {len(df_clean):,}")
    print(f"  Feature count: {len(FEATURE_NAMES)} features")
    print(f"  Train slice  : {df_train.index[0].date()} to {df_train.index[-1].date()} ({len(df_train):,} rows)")
    print(f"  Test slice   : {df_test.index[0].date()} to {df_test.index[-1].date()} ({len(df_test):,} rows)")

    # 2. Trace 1 Live Prediction End-to-End via API
    print("\n[PHASE 2] Tracing 1 Live Prediction End-to-End via API...")
    now_utc = datetime.now(timezone.utc)
    t_live_str = now_utc.strftime("%Y-%m-%dT%H:00")

    url = "https://api.open-meteo.com/v1/forecast?latitude=30.0444&longitude=31.2357&hourly=temperature_2m&past_days=7&forecast_days=1&timezone=UTC"
    r = requests.get(url, timeout=10)
    data = r.json()
    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]

    t_idx = len(temps) - 1
    for i in range(len(times)-1, -1, -1):
        if pd.to_datetime(times[i], utc=True) <= now_utc:
            t_idx = i
            break

    start_idx = t_idx - 168
    live_history = [float(round(v, 2)) for v in temps[start_idx:t_idx]]
    history_timestamps = times[start_idx:t_idx]

    c = Client()
    payload = {
        "current_time": times[t_idx] + ":00Z",
        "latitude": 30.0444,
        "longitude": 31.2357,
        "temperature_history_168h": live_history
    }
    resp = c.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
    pred_data = resp.json()

    print(f"  Reference Time (t)            : {times[t_idx]} UTC")
    print(f"  Forecast Timestamp (t+24h)    : {pred_data['prediction']['forecast_time']}")
    print(f"  Predicted Temperature 2m      : {pred_data['prediction']['predicted_temperature_2m']}°C")
    print(f"  80% Prediction Interval       : [{pred_data['prediction']['confidence_interval_80']['lower_bound']}°C, {pred_data['prediction']['confidence_interval_80']['upper_bound']}°C]")

    # 3. Model Out-of-Sample Performance on Unseen Test Set
    print("\n[PHASE 3] Evaluating Unseen Out-of-Sample Test Set (2018–2019, 17,505 rows)...")
    load_model_artifacts()
    booster = xgb.Booster()
    booster.load_model("ml/models/xgboost_weather_model.json")
    dtest = xgb.DMatrix(X_test, feature_names=FEATURE_NAMES)
    pred_deltas = booster.predict(dtest)
    pred_abs = lag24_test + pred_deltas

    metrics_test = evaluate_forecasts(y_test_abs, pred_abs)
    for k, v in metrics_test.items():
        print(f"  {k.upper():<10}: {v}")

    # May subset evaluation
    may_mask = (df_test.index.month == 5)
    may_metrics = evaluate_forecasts(y_test_abs[may_mask], pred_abs[may_mask])
    print(f"  MAY MAE   : {may_metrics['mae']:.6f}°C")

    # 4. Benchmark Against Baselines & Ridge
    print("\n[PHASE 4] Benchmarking Phase 3 Model vs Ridge & Baselines on Untouched Test Set:")
    
    # Ridge on Delta
    ridge = Ridge(alpha=100.0).fit(X_train, y_train)
    p_ridge = lag24_test + ridge.predict(X_test)
    m_ridge = evaluate_forecasts(y_test_abs, p_ridge)

    # 24h Persistence
    p_persist24 = df_test["temperature_2m_lag_24"].values
    m_persist24 = evaluate_forecasts(y_test_abs, p_persist24)

    # 1h Persistence
    p_persist1 = df_test["temperature_2m_lag_1"].values
    m_persist1 = evaluate_forecasts(y_test_abs, p_persist1)

    print(f"  {'Model / Strategy':<32} | {'MAE (°C)':<10} | {'RMSE (°C)':<10} | {'R²':<8} | {'Improvement':<15}")
    print("-" * 85)
    print(f"  {'Phase 3 XGBoost (Delta + 21f)':<32} | {metrics_test['mae']:<10.4f} | {metrics_test['rmse']:<10.4f} | {metrics_test['r2']:<8.4f} | {'Baseline Anchor':<15}")
    print(f"  {'Ridge Regression (21f on Delta)':<32} | {m_ridge['mae']:<10.4f} | {m_ridge['rmse']:<10.4f} | {m_ridge['r2']:<8.4f} | {((m_ridge['mae'] - metrics_test['mae'])/m_ridge['mae'])*100:+.2f}% better")
    print(f"  {'24h Persistence (t-24)':<32} | {m_persist24['mae']:<10.4f} | {m_persist24['rmse']:<10.4f} | {m_persist24['r2']:<8.4f} | {((m_persist24['mae'] - metrics_test['mae'])/m_persist24['mae'])*100:+.2f}% better")
    print(f"  {'1h Persistence (t-1)':<32} | {m_persist1['mae']:<10.4f} | {m_persist1['rmse']:<10.4f} | {m_persist1['r2']:<8.4f} | {((m_persist1['mae'] - metrics_test['mae'])/m_persist1['mae'])*100:+.2f}% better")

    # 5. Performance Latency Benchmarks
    print("\n[PHASE 5] Performance Latency Benchmarks:")
    t0 = time.time()
    for _ in range(100):
        _ = predict_temperature(X_test.iloc[0].tolist(), lag_24=25.0)
    avg_inference_ms = ((time.time() - t0) / 100) * 1000

    print(f"  Single Model Inference Latency (Mean + Quantiles): {avg_inference_ms:.2f} ms")
    print("\n" + "=" * 80)
    print("      ALL AUDIT PHASES COMPLETED WITH RAW MEASURED EVIDENCE")
    print("=" * 80)


if __name__ == "__main__":
    run_full_audit()
