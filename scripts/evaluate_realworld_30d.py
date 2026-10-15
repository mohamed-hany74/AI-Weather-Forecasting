"""
Step 5: Real-World Evaluation Script for WeatherCast AI.
Benchmarks 30+ days (2026-08-01 to 2026-09-30, 61 days) comparing:
(1) Production-like inputs (Historical Forecast API + calibration)
(2) ERA5-style inputs (Archive API)
against observed ERA5 ground truth at t+24h.
Evaluates Cairo (in-region) and Alexandria (out-of-region baseline).
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from weather.domain.contracts import FEATURE_NAMES, EXOGENOUS_CALIBRATION_OFFSETS
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features
from weather.services.weather_service import calibrate_exogenous
from weather.services.model_service import load_model_artifacts, predict_temperature
from sklearn.metrics import mean_absolute_error, mean_squared_error


def fetch_evaluation_stream(lat: float, lon: float, start_date: str, end_date: str, source: str = "forecast"):
    """
    Fetch atmospheric sequence for evaluation.
    source='forecast': queries historical-forecast-api.open-meteo.com
    source='archive': queries archive-api.open-meteo.com
    """
    base_url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        if source == "forecast"
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    vars_list = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_10m",
        "cloud_cover",
        "pressure_msl"
    ]
    url = (
        f"{base_url}?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(vars_list)}&timezone=UTC"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df


def evaluate_location(name: str, lat: float, lon: float, start_date: str, end_date: str, is_out_of_region: bool = False):
    print(f"\n{'='*70}")
    print(f"Evaluating {name} ({'OUT-OF-REGION BENCHMARK' if is_out_of_region else 'TRAINING REGION'})")
    print(f"Coordinates: lat={lat}, lon={lon} | Period: {start_date} to {end_date}")
    print(f"{'='*70}")

    # Fetch ground truth (ERA5 observed archive)
    df_truth = fetch_evaluation_stream(lat, lon, start_date, end_date, source="archive")
    # Fetch production-like inputs (historical forecast API)
    df_fore = fetch_evaluation_stream(lat, lon, start_date, end_date, source="forecast")

    # Align timestamps
    common_idx = df_truth.index.intersection(df_fore.index)
    df_truth = df_truth.loc[common_idx]
    df_fore = df_fore.loc[common_idx]

    load_model_artifacts()

    # We need 168h prior to t, and ground truth at t+24h.
    # Therefore, valid reference hours t run from index[168] to index[-25].
    eval_timestamps = common_idx[168:-24]
    print(f"Total evaluation instances (forecast steps): {len(eval_timestamps)}")

    results = {
        "production_like": {"y_true": [], "y_pred": [], "timestamps": [], "errors": []},
        "era5_style": {"y_true": [], "y_pred": [], "timestamps": [], "errors": []}
    }

    for i, t in enumerate(eval_timestamps):
        idx_in_common = 168 + i
        t_target = t + timedelta(hours=24)
        y_true_val = float(df_truth.loc[t_target, "temperature_2m"])

        # 1. Production-like evaluation:
        # History is past 168h from forecast/observed stream [t-168 to t-1]
        hist_168_prod = df_fore["temperature_2m"].iloc[idx_in_common - 168 : idx_in_common].tolist()
        row_fore_t = df_fore.iloc[idx_in_common]
        row_fore_p3 = df_fore.iloc[idx_in_common - 3]
        row_fore_p6 = df_fore.iloc[idx_in_common - 6]

        p_ch3_prod = float(row_fore_t["pressure_msl"] - row_fore_p3["pressure_msl"])
        p_ch6_prod = float(row_fore_t["pressure_msl"] - row_fore_p6["pressure_msl"])

        exo_prod = calibrate_exogenous(
            apparent_t=float(row_fore_t["apparent_temperature"]),
            press_msl=float(row_fore_t["pressure_msl"]),
            rel_hum=float(row_fore_t["relative_humidity_2m"]),
            dew=float(row_fore_t["dew_point_2m"]),
            wind=float(row_fore_t["wind_speed_10m"]),
            cloud=float(row_fore_t["cloud_cover"]),
            p_ch3=p_ch3_prod,
            p_ch6=p_ch6_prod
        )

        feats_prod = build_production_features(hist_168_prod, t, exo_prod)
        pred_prod = predict_temperature(feats_prod, lag_24=float(hist_168_prod[-24]))
        err_prod = pred_prod - y_true_val

        results["production_like"]["y_true"].append(y_true_val)
        results["production_like"]["y_pred"].append(pred_prod)
        results["production_like"]["timestamps"].append(t)
        results["production_like"]["errors"].append(err_prod)

        # 2. ERA5-style evaluation:
        hist_168_era5 = df_truth["temperature_2m"].iloc[idx_in_common - 168 : idx_in_common].tolist()
        row_era5_t = df_truth.iloc[idx_in_common]
        row_era5_p3 = df_truth.iloc[idx_in_common - 3]
        row_era5_p6 = df_truth.iloc[idx_in_common - 6]

        p_ch3_era5 = float(row_era5_t["pressure_msl"] - row_era5_p3["pressure_msl"])
        p_ch6_era5 = float(row_era5_t["pressure_msl"] - row_era5_p6["pressure_msl"])

        exo_era5 = ExogenousWeather(
            apparent_temperature=float(row_era5_t["apparent_temperature"]),
            pressure_msl=float(row_era5_t["pressure_msl"]),
            relative_humidity_2m=float(row_era5_t["relative_humidity_2m"]),
            dew_point_2m=float(row_era5_t["dew_point_2m"]),
            wind_speed_10m=float(row_era5_t["wind_speed_10m"]),
            cloud_cover=float(row_era5_t["cloud_cover"]),
            pressure_change_3h=p_ch3_era5,
            pressure_change_6h=p_ch6_era5
        )

        feats_era5 = build_production_features(hist_168_era5, t, exo_era5)
        pred_era5 = predict_temperature(feats_era5, lag_24=float(hist_168_era5[-24]))
        err_era5 = pred_era5 - y_true_val

        results["era5_style"]["y_true"].append(y_true_val)
        results["era5_style"]["y_pred"].append(pred_era5)
        results["era5_style"]["timestamps"].append(t)
        results["era5_style"]["errors"].append(err_era5)

    summary = {"location": name, "is_out_of_region": is_out_of_region, "modes": {}}

    for mode in ["production_like", "era5_style"]:
        y_t = np.array(results[mode]["y_true"])
        y_p = np.array(results[mode]["y_pred"])
        errs = np.array(results[mode]["errors"])
        abs_errs = np.abs(errs)

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        mean_bias = float(np.mean(errs))
        max_error = float(np.max(abs_errs))

        # Worst 5 days by daily average MAE
        df_daily = pd.DataFrame({
            "timestamp": results[mode]["timestamps"],
            "abs_error": abs_errs,
            "error": errs,
            "y_true": y_t,
            "y_pred": y_p
        })
        df_daily["day"] = df_daily["timestamp"].dt.date
        worst_days = (
            df_daily.groupby("day")["abs_error"]
            .mean()
            .reset_index()
            .sort_values(by="abs_error", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )
        for w in worst_days:
            w["day"] = str(w["day"])
            w["abs_error"] = round(float(w["abs_error"]), 4)

        summary["modes"][mode] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_bias": round(mean_bias, 4),
            "max_error": round(max_error, 4),
            "worst_5_days": worst_days
        }

        print(f"\n--- Results for {name} ({mode}) ---")
        print(f"  MAE       : {mae:.4f} °C")
        print(f"  RMSE      : {rmse:.4f} °C")
        print(f"  Mean Bias : {mean_bias:+.4f} °C")
        print(f"  Max Error : {max_error:.4f} °C")
        print(f"  Worst 5 Days:")
        for idx_w, w in enumerate(worst_days, 1):
            print(f"    {idx_w}. {w['day']}: Daily MAE = {w['abs_error']:.2f} °C")

    return summary


def main():
    start_date = "2026-08-01"
    end_date = "2026-09-30"

    cairo_summary = evaluate_location("Cairo", 30.0444, 31.2357, start_date, end_date, is_out_of_region=False)
    alex_summary = evaluate_location("Alexandria", 31.2001, 29.9187, start_date, end_date, is_out_of_region=True)

    output = {
        "evaluation_period": f"{start_date} to {end_date} (61 days, August-September 2026)",
        "ground_truth_source": "Open-Meteo Archive ERA5 Reanalysis (Observed actuals at t+24h)",
        "production_source": "Open-Meteo Historical Forecast API (operational model inputs) + EXOGENOUS_CALIBRATION_OFFSETS",
        "era5_source": "Open-Meteo Archive ERA5 Reanalysis (reanalysis inputs)",
        "cairo": cairo_summary,
        "alexandria": alex_summary
    }

    report_dir = os.path.join(PROJECT_ROOT, "artifacts", "phase5a")
    os.makedirs(report_dir, exist_ok=True)
    report_json_path = os.path.join(report_dir, "step5_realworld_evaluation.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Saved Step 5 evaluation results to {report_json_path}")


if __name__ == "__main__":
    main()
