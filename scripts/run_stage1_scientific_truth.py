"""
Scientific Truth Evaluation Script for WeatherCast AI (Stage 1).

Executes:
1. Serving Skew Evaluation:
   - Fetches 67 days (2026-07-29 to 2026-10-03) of Cairo atmospheric data from Open-Meteo Forecast API.
   - Fetches identical period from Open-Meteo Archive ERA5 reanalysis.
   - Computes distribution divergence and deltas across atmospheric variables:
     (apparent_temperature, relative_humidity_2m, pressure_msl).
   - Feeds both sources through the production XGBoost model to evaluate impact on predicted temperature.
2. Open-Meteo NWP Forecast Baseline vs Model:
   - Benchmarks actual observed temperatures vs Open-Meteo NWP forecasts (Blend & GFS).
   - Compares with our XGBoost model predictions.
3. Rolling-Origin Cross-Validation (2014-2019):
   - Trains iteratively on all preceding years (with 24h purge gap) and tests on the next full year.
   - Benchmarks XGBoost vs Ridge regression on the identical 15 causal features.
   - Computes Stationary Block Bootstrap (block size = 168 hours / 7 days) on the difference in absolute errors.
   - Calculates 95% Confidence Intervals and hypothesis test p-value.
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from weather.domain.contracts import FEATURE_NAMES
from ml.training.features import engineer_dataframe_features
from weather.services.model_service import load_model_artifacts

LAT = 30.0444
LON = 31.2357
START_DATE = "2026-07-29"
END_DATE = "2026-10-03"

def run_serving_skew_analysis(booster):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: SERVING SKEW ANALYSIS (Open-Meteo Forecast vs ERA5)")
    print("=" * 80)
    print(f"Sampling period: {START_DATE} to {END_DATE} (67 days, 1608 hours)")

    # 1. Fetch Forecast API (simulates runtime serving input)
    print("[1/4] Fetching Open-Meteo Forecast API historical runs...")
    url_forecast = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}&models=gfs_seamless"
        f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,pressure_msl"
        f"&timezone=UTC"
    )
    r_f = requests.get(url_forecast, timeout=25)
    r_f.raise_for_status()
    df_f = pd.DataFrame(r_f.json()["hourly"])
    df_f["time"] = pd.to_datetime(df_f["time"], utc=True)
    df_f.set_index("time", inplace=True)

    # 2. Fetch Archive ERA5 (training distribution source)
    print("[2/4] Fetching Open-Meteo Archive ERA5 reanalysis...")
    url_archive = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}&models=era5"
        f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,pressure_msl"
        f"&timezone=UTC"
    )
    r_a = requests.get(url_archive, timeout=25)
    r_a.raise_for_status()
    df_a = pd.DataFrame(r_a.json()["hourly"])
    df_a["time"] = pd.to_datetime(df_a["time"], utc=True)
    df_a.set_index("time", inplace=True)

    # Align timestamps
    common_index = df_f.index.intersection(df_a.index)
    df_f = df_f.loc[common_index].copy()
    df_a = df_a.loc[common_index].copy()
    print(f"Aligned hours: {len(common_index)}")

    # Variable differences
    vars_to_compare = ["apparent_temperature", "relative_humidity_2m", "pressure_msl", "temperature_2m"]
    diff_metrics = {}
    print("\n--- Distribution Deltas: Forecast API minus ERA5 ---")
    for v in vars_to_compare:
        delta = df_f[v] - df_a[v]
        mae_v = np.mean(np.abs(delta))
        mean_v = np.mean(delta)
        std_v = np.std(delta)
        max_abs = np.max(np.abs(delta))
        diff_metrics[v] = {
            "mean_error": float(mean_v),
            "mae": float(mae_v),
            "std": float(std_v),
            "max_abs": float(max_abs),
            "era5_mean": float(df_a[v].mean()),
            "fore_mean": float(df_f[v].mean())
        }
        print(f"  {v:<22}: MAE={mae_v:.3f}, Bias={mean_v:+.3f}, Std={std_v:.3f}, MaxAbs={max_abs:.3f}")

    # Build features using ERA5 vs Forecast
    print("\n[3/4] Passing both atmospheric streams through Model feature pipeline...")
    # Engineer features on both series
    feat_era5 = engineer_dataframe_features(df_a)
    feat_fore = engineer_dataframe_features(df_f)

    valid_mask = feat_era5[FEATURE_NAMES].notna().all(axis=1) & feat_fore[FEATURE_NAMES].notna().all(axis=1)
    eval_index = feat_era5[valid_mask].index
    print(f"Usable evaluation timestamps with full 168h history: {len(eval_index)}")

    if len(eval_index) > 0:
        dmat_era5 = xgb.DMatrix(feat_era5.loc[eval_index, FEATURE_NAMES])
        dmat_fore = xgb.DMatrix(feat_fore.loc[eval_index, FEATURE_NAMES])

        pred_from_era5 = booster.predict(dmat_era5)
        pred_from_fore = booster.predict(dmat_fore)

        pred_delta = pred_from_fore - pred_from_era5
        pred_mae = np.mean(np.abs(pred_delta))
        pred_bias = np.mean(pred_delta)
        pred_max = np.max(np.abs(pred_delta))

        # Compare both against actual ground truth (target t+24 from ERA5 observations)
        if "target" in feat_era5.columns and feat_era5.loc[eval_index, "target"].notna().all():
            y_true = feat_era5.loc[eval_index, "target"].values
            mae_era5_true = mean_absolute_error(y_true, pred_from_era5)
            mae_fore_true = mean_absolute_error(y_true, pred_from_fore)
        else:
            y_true = None
            mae_era5_true = None
            mae_fore_true = None

        print("\n--- Model Output Sensitivity to Serving Distribution Shift ---")
        print(f"  Prediction delta (Forecast - ERA5) MAE : {pred_mae:.4f}°C")
        print(f"  Prediction delta Mean Bias              : {pred_bias:+.4f}°C")
        print(f"  Max Prediction Discrepancy             : {pred_max:.4f}°C")
        if mae_era5_true is not None:
            print(f"  Actual Forecast MAE using ERA5 features: {mae_era5_true:.4f}°C")
            print(f"  Actual Forecast MAE using Forecast feed: {mae_fore_true:.4f}°C")
    else:
        pred_mae, pred_bias, pred_max = 0, 0, 0
        mae_era5_true, mae_fore_true = None, None

    return {
        "diff_metrics": diff_metrics,
        "prediction_impact": {
            "pred_mae": float(pred_mae),
            "pred_bias": float(pred_bias),
            "pred_max": float(pred_max),
            "mae_era5_true": float(mae_era5_true) if mae_era5_true else None,
            "mae_fore_true": float(mae_fore_true) if mae_fore_true else None
        },
        "df_f": df_f,
        "df_a": df_a
    }


def run_nwp_benchmark(booster, df_f, df_a):
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: OPEN-METEO NWP BENCHMARK vs WEATHERCAST XGBOOST (t + 24h)")
    print("=" * 80)
    
    # Ground truth: actual observed temperature_2m from ERA5 reanalysis at t + 24h
    # NWP prediction for t + 24h:
    # Open-Meteo GFS forecast generated for t+24h vs our XGBoost
    
    # In df_f: 'temperature_2m' (Blend) and 'temperature_2m_gfs_seamless' (GFS)
    # Ground truth at timestamp tau is df_a['temperature_2m'].loc[tau]
    # For a forecast made at time t for horizon t+24h:
    # A true 24h ahead forecast evaluates against actual at t+24h.
    
    df_eval = pd.DataFrame(index=df_a.index)
    df_eval["actual"] = df_a["temperature_2m"]
    df_eval["actual_t_plus_24"] = df_a["temperature_2m"].shift(-24)
    
    # Baseline 1: Open-Meteo forecast at t+24
    df_eval["nwp_forecast_t_plus_24"] = df_f["temperature_2m"].shift(-24)
    if "temperature_2m_gfs_seamless" in df_f.columns:
        df_eval["gfs_forecast_t_plus_24"] = df_f["temperature_2m_gfs_seamless"].shift(-24)
    
    # Baseline 2: Persistence from t-1 to t+24
    df_eval["persist_t_minus_1"] = df_a["temperature_2m"].shift(1)
    # Baseline 3: Same-hour persistence t-24 to t+24
    df_eval["persist_same_hour"] = df_a["temperature_2m"].shift(0)  # temp at t predicting t+24
    
    # Model predictions
    feat_fore = engineer_dataframe_features(df_f)
    valid_idx = feat_fore.dropna(subset=FEATURE_NAMES).index
    valid_idx = valid_idx.intersection(df_eval.dropna(subset=["actual_t_plus_24"]).index)
    
    dmat = xgb.DMatrix(feat_fore.loc[valid_idx, FEATURE_NAMES])
    df_eval.loc[valid_idx, "xgb_pred"] = booster.predict(dmat)
    
    clean_bench = df_eval.dropna(subset=["actual_t_plus_24", "xgb_pred", "nwp_forecast_t_plus_24"])
    print(f"Evaluated 24h prediction horizons: {len(clean_bench)} hours")
    
    y_true = clean_bench["actual_t_plus_24"].values
    
    metrics = {}
    models_to_test = {
        "WeatherCast XGBoost (Ours)": clean_bench["xgb_pred"].values,
        "Open-Meteo Blend NWP": clean_bench["nwp_forecast_t_plus_24"].values,
        "Persistence from t (24h lead)": clean_bench.loc[clean_bench.index, "persist_same_hour"].values,
        "Persistence from t-1h (25h lead)": clean_bench.loc[clean_bench.index, "persist_t_minus_1"].values,
    }
    if "gfs_forecast_t_plus_24" in clean_bench.columns and clean_bench["gfs_forecast_t_plus_24"].notna().all():
        models_to_test["Open-Meteo GFS NWP"] = clean_bench["gfs_forecast_t_plus_24"].values

    print("\n--- 24-Hour Horizon Benchmark Results ---")
    print(f"{'Strategy / Model':<35} | {'MAE (°C)':<10} | {'RMSE (°C)':<10} | {'R²':<8} | {'MedAE (°C)':<10}")
    print("-" * 82)
    for name, preds in models_to_test.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        medae = np.median(np.abs(y_true - preds))
        metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2, "medae": medae}
        print(f"{name:<35} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<8.4f} | {medae:<10.4f}")

    return metrics


def run_rolling_origin_cv_with_bootstrap():
    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: ROLLING-ORIGIN CV (2014-2019) + BLOCK BOOTSTRAP TEST")
    print("=" * 80)

    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    df_feat = engineer_dataframe_features(df)
    df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target"]).copy()

    test_years = range(2014, 2020)
    cv_records = []
    
    all_abs_errors_xgb = []
    all_abs_errors_ridge = []
    
    print(f"{'Test Year':<10} | {'Train Range':<22} | {'Test Size':<10} | {'XGB MAE':<10} | {'Ridge MAE':<10} | {'Delta (XGB-Ridge)':<18}")
    print("-" * 92)

    for test_year in test_years:
        test_start = pd.to_datetime(f"{test_year}-01-01 00:00:00+00:00")
        purge_cutoff = test_start - pd.Timedelta(hours=24)
        
        train_mask = df_clean.index < purge_cutoff
        test_mask = df_clean.index.year == test_year
        
        df_tr = df_clean[train_mask]
        df_te = df_clean[test_mask]
        
        X_tr, y_tr = df_tr[FEATURE_NAMES], df_tr["target"]
        X_te, y_te = df_te[FEATURE_NAMES], df_te["target"]
        
        # Train XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        model_xgb.fit(X_tr, y_tr)
        pred_xgb = model_xgb.predict(X_te)
        
        # Train Ridge
        model_ridge = Ridge(alpha=1.0)
        model_ridge.fit(X_tr, y_tr)
        pred_ridge = model_ridge.predict(X_te)
        
        err_xgb = np.abs(y_te.values - pred_xgb)
        err_ridge = np.abs(y_te.values - pred_ridge)
        
        mae_xgb = np.mean(err_xgb)
        mae_ridge = np.mean(err_ridge)
        delta = mae_xgb - mae_ridge
        
        all_abs_errors_xgb.extend(err_xgb)
        all_abs_errors_ridge.extend(err_ridge)
        
        train_range_str = f"2010–{test_year-1}"
        print(f"{test_year:<10} | {train_range_str:<22} | {len(y_te):<10} | {mae_xgb:<10.4f} | {mae_ridge:<10.4f} | {delta:<+18.4f}")
        
        cv_records.append({
            "test_year": test_year,
            "train_range": train_range_str,
            "n_test": len(y_te),
            "mae_xgb": mae_xgb,
            "mae_ridge": mae_ridge,
            "delta": delta,
            "rmse_xgb": np.sqrt(mean_squared_error(y_te, pred_xgb)),
            "rmse_ridge": np.sqrt(mean_squared_error(y_te, pred_ridge)),
            "r2_xgb": r2_score(y_te, pred_xgb),
            "r2_ridge": r2_score(y_te, pred_ridge)
        })

    cv_df = pd.DataFrame(cv_records)
    mean_xgb = cv_df["mae_xgb"].mean()
    mean_ridge = cv_df["mae_ridge"].mean()
    mean_delta = cv_df["delta"].mean()
    print("-" * 92)
    print(f"{'MEAN':<10} | {'--':<22} | {len(all_abs_errors_xgb):<10} | {mean_xgb:<10.4f} | {mean_ridge:<10.4f} | {mean_delta:<+18.4f}")

    # Stationary / Circular Block Bootstrap for Time Series
    # Block size = 168 hours (7 days / 1 full weekly cycle)
    print("\n--- Moving Block Bootstrap (1,000 resamples, Block Size = 168h) ---")
    np.random.seed(42)
    arr_xgb = np.array(all_abs_errors_xgb)
    arr_ridge = np.array(all_abs_errors_ridge)
    N = len(arr_xgb)
    block_size = 168
    num_blocks = int(np.ceil(N / block_size))
    n_boot = 1000

    diff_series = arr_xgb - arr_ridge  # positive means XGBoost has HIGHER error (worse)
    observed_mean_diff = np.mean(diff_series)

    boot_mean_diffs = []
    for _ in range(n_boot):
        # Sample starting indices of blocks
        start_indices = np.random.randint(0, N - block_size + 1, size=num_blocks)
        indices = np.concatenate([np.arange(idx, idx + block_size) for idx in start_indices])[:N]
        boot_diff = np.mean(diff_series[indices])
        boot_mean_diffs.append(boot_diff)

    boot_mean_diffs = np.array(boot_mean_diffs)
    ci_lower = np.percentile(boot_mean_diffs, 2.5)
    ci_upper = np.percentile(boot_mean_diffs, 97.5)
    
    # Two-sided empirical p-value for H0: mean_diff == 0
    # Fraction of bootstrap distribution crossing 0 in opposite direction
    if observed_mean_diff > 0:
        p_val = 2 * np.mean(boot_mean_diffs <= 0)
    else:
        p_val = 2 * np.mean(boot_mean_diffs >= 0)
    p_val = min(p_val, 1.0)

    print(f"Observed Delta MAE (XGB - Ridge): {observed_mean_diff:+.5f}°C")
    print(f"95% Block Bootstrap CI          : [{ci_lower:+.5f}°C, {ci_upper:+.5f}°C]")
    print(f"Empirical p-value (H0: diff = 0): {p_val:.4f}")
    if ci_lower > 0:
        conclusion = "Ridge is STATISTICALLY SIGNIFICANTLY BETTER than XGBoost (p < 0.05)"
    elif ci_upper < 0:
        conclusion = "XGBoost is STATISTICALLY SIGNIFICANTLY BETTER than Ridge (p < 0.05)"
    else:
        conclusion = "NO STATISTICALLY SIGNIFICANT DIFFERENCE between XGBoost and Ridge (CI spans 0)"
    print(f"Conclusion                      : {conclusion}")

    return {
        "cv_df": cv_df,
        "mean_xgb": float(mean_xgb),
        "mean_ridge": float(mean_ridge),
        "observed_mean_diff": float(observed_mean_diff),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "p_value": float(p_val),
        "conclusion": conclusion
    }


def main():
    print("=" * 80)
    print("  STAGE 1: THE SCIENTIFIC TRUTH EMPIRICAL AUDIT")
    print("=" * 80)
    
    # Load frozen production model
    booster = xgb.Booster()
    model_path = os.path.join(PROJECT_ROOT, "ml", "models", "xgboost_weather_model.json")
    booster.load_model(model_path)
    
    # 1. Serving Skew
    skew_res = run_serving_skew_analysis(booster)
    
    # 2. NWP Baseline
    nwp_res = run_nwp_benchmark(booster, skew_res["df_f"], skew_res["df_a"])
    
    # 3. Rolling-Origin CV + Bootstrap
    cv_res = run_rolling_origin_cv_with_bootstrap()
    
    # Save raw results
    output_path = os.path.join(PROJECT_ROOT, "ml", "training", "scientific_truth_audit_results.json")
    serializable = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "skew_analysis": {
            "diff_metrics": skew_res["diff_metrics"],
            "prediction_impact": skew_res["prediction_impact"]
        },
        "nwp_benchmark": {k: {m: float(v) for m, v in vals.items()} for k, vals in nwp_res.items()},
        "rolling_origin_cv": {
            "cv_table": cv_res["cv_df"].to_dict(orient="records"),
            "mean_xgb": cv_res["mean_xgb"],
            "mean_ridge": cv_res["mean_ridge"],
            "observed_mean_diff": cv_res["observed_mean_diff"],
            "ci_95": cv_res["ci_95"],
            "p_value": cv_res["p_value"],
            "conclusion": cv_res["conclusion"]
        }
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"\n[DONE] Saved complete empirical results to: {output_path}")

if __name__ == "__main__":
    main()
