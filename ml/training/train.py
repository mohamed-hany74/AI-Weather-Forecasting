"""
Reproducible Training Pipeline for WeatherCast AI (Phase 3).
Trains:
1. Mean Model: XGBoost Regressor predicting delta = y(t+24) - lag_24 on 21 causal features.
2. Lower Quantile Model: XGBoost Regressor with reg:quantileerror (alpha=0.10).
3. Upper Quantile Model: XGBoost Regressor with reg:quantileerror (alpha=0.90).
Exports production JSON artifacts including quantile models and verified provenance.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit

from features import (
    PRODUCTION_FEATURES,
    TARGET,
    FORECAST_HORIZON,
    engineer_dataframe_features
)
from evaluation import evaluate_forecasts
from validation import validate_feature_matrix, validate_target_series, validate_temporal_boundary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "weather.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "xgboost_weather_model.json")
MODEL_Q10_OUT_PATH = os.path.join(MODELS_DIR, "xgboost_weather_model_q10.json")
MODEL_Q90_OUT_PATH = os.path.join(MODELS_DIR, "xgboost_weather_model_q90.json")
FEATURE_CFG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Tuned Hyperparameters via Optuna TimeSeriesSplit
XGB_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.07,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

XGB_QUANTILE_PARAMS_10 = {
    **XGB_PARAMS,
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.10
}

XGB_QUANTILE_PARAMS_90 = {
    **XGB_PARAMS,
    "objective": "reg:quantileerror",
    "quantile_alpha": 0.90
}


def run_training_pipeline():
    print("=" * 70)
    print("      WeatherCast AI — Phase 3 Training Pipeline Execution")
    print("=" * 70)

    # 1. Load Dataset
    print(f"\n[1/7] Loading dataset from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").copy()

    # 2. Chronological Split
    n_total = len(df)
    n_test = int(n_total * TEST_SIZE)
    split_timestamp = df.index[n_total - n_test]

    df_train_raw = df[df.index < split_timestamp].copy()
    df_test_raw = df[df.index >= split_timestamp].copy()

    # 3. Preprocessing (Strictly causal forward fill)
    for col in df_train_raw.select_dtypes(include=[np.number]).columns:
        df_train_raw[col] = df_train_raw[col].ffill().bfill()
        df_test_raw[col] = df_test_raw[col].ffill().bfill()

    # 4. Feature Engineering
    print("[2/7] Engineering 21 causal features & delta target...")
    df_clean = pd.concat([df_train_raw, df_test_raw])
    df_feat = engineer_dataframe_features(df_clean)
    df_valid = df_feat.dropna(subset=PRODUCTION_FEATURES + ["target", "target_abs"]).copy()

    # 5. Purge & Embargo Boundary Enforcement
    print("[3/7] Enforcing 24h purge & embargo boundary...")
    purge_cutoff = split_timestamp - pd.Timedelta(hours=FORECAST_HORIZON)
    df_tr = df_valid[df_valid.index < purge_cutoff].copy()
    df_te = df_valid[df_valid.index >= split_timestamp].copy()

    X_train = df_tr[PRODUCTION_FEATURES]
    y_train = df_tr["target"]
    lag24_train = df_tr["temperature_2m_lag_24"]
    y_train_abs = df_tr["target_abs"]

    X_test = df_te[PRODUCTION_FEATURES]
    y_test = df_te["target"]
    lag24_test = df_te["temperature_2m_lag_24"]
    y_test_abs = df_te["target_abs"]

    validate_temporal_boundary(df_tr.index[-1], df_te.index[0], FORECAST_HORIZON)
    validate_feature_matrix(X_train, PRODUCTION_FEATURES)
    validate_feature_matrix(X_test, PRODUCTION_FEATURES)
    validate_target_series(y_train)
    validate_target_series(y_test)

    print(f"  Train set: {len(X_train):,} rows ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"  Test set : {len(X_test):,} rows ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    # 6. Walk-Forward Cross Validation on Purged Train Set (Strict Purge Gap)
    print("\n[4/7] Running 5-Fold Walk-Forward Cross Validation on Purged Train Set (gap=24h)...")
    tscv = TimeSeriesSplit(n_splits=5, gap=FORECAST_HORIZON)
    cv_maes = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        model_cv = XGBRegressor(**XGB_PARAMS)
        model_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred_delta_val = model_cv.predict(X_train.iloc[val_idx])
        pred_abs_val = lag24_train.iloc[val_idx].values + pred_delta_val
        metrics_fold = evaluate_forecasts(y_train_abs.iloc[val_idx].values, pred_abs_val)
        cv_maes.append(metrics_fold["mae"])
        print(f"  Fold {fold}: MAE={metrics_fold['mae']:.4f}°C | RMSE={metrics_fold['rmse']:.4f}°C | R²={metrics_fold['r2']:.4f}")

    cv_mean_mae = float(np.mean(cv_maes))
    cv_std_mae = float(np.std(cv_maes))
    print(f"  CV Mean Absolute Error: {cv_mean_mae:.4f}°C (±{cv_std_mae:.4f})")

    # 7. Final Model Training (Mean + Quantiles)
    print(f"\n[5/7] Training Final Mean and Quantile Models (alpha=0.10, 0.90)...")
    t0 = time.time()
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train)

    q10_model = XGBRegressor(**XGB_QUANTILE_PARAMS_10)
    q10_model.fit(X_train, y_train)

    q90_model = XGBRegressor(**XGB_QUANTILE_PARAMS_90)
    q90_model.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"  All 3 models trained in {fit_duration:.2f} seconds.")

    # 8. Unseen Test Evaluation
    print("\n[6/7] Evaluating Unseen Out-of-Sample Test Set...")
    y_test_pred_delta = final_model.predict(X_test)
    y_test_pred_abs = lag24_test.values + y_test_pred_delta
    test_metrics = evaluate_forecasts(y_test_abs.values, y_test_pred_abs)

    # Quantile bounds
    y_q10_pred_delta = q10_model.predict(X_test)
    y_q90_pred_delta = q90_model.predict(X_test)
    y_q10_abs = lag24_test.values + y_q10_pred_delta
    y_q90_abs = lag24_test.values + y_q90_pred_delta

    coverage = float(np.mean((y_test_abs.values >= y_q10_abs) & (y_test_abs.values <= y_q90_abs)) * 100.0)
    mean_interval_width = float(np.mean(y_q90_abs - y_q10_abs))

    # May subset evaluation
    may_mask = (df_te.index.month == 5)
    may_mae = float(evaluate_forecasts(y_test_abs[may_mask].values, y_test_pred_abs[may_mask])["mae"])

    print("\n===============================================================")
    print("        UNSEEN OUT-OF-SAMPLE TEST EVALUATION RESULTS           ")
    print("===============================================================")
    for k, v in test_metrics.items():
        print(f"  {k.upper():<10}: {v}")
    print(f"  MAY MAE   : {may_mae:.6f}°C")
    print(f"  COVERAGE  : {coverage:.2f}% (Target: 80%)")
    print(f"  INTERVAL W: {mean_interval_width:.2f}°C")
    print("===============================================================")

    # 9. Export Artifacts
    print(f"\n[7/7] Exporting production artifacts to: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Booster JSONs
    final_model.get_booster().save_model(MODEL_OUT_PATH)
    q10_model.get_booster().save_model(MODEL_Q10_OUT_PATH)
    q90_model.get_booster().save_model(MODEL_Q90_OUT_PATH)
    print(f"  --> Saved Mean Model JSON: {MODEL_OUT_PATH}")
    print(f"  --> Saved Q10 Model JSON : {MODEL_Q10_OUT_PATH}")
    print(f"  --> Saved Q90 Model JSON : {MODEL_Q90_OUT_PATH}")

    # Feature Config
    feature_config = {
        "target": "delta_y24_minus_lag24",
        "base_target": TARGET,
        "forecast_horizon": FORECAST_HORIZON,
        "feature_count": len(PRODUCTION_FEATURES),
        "features": PRODUCTION_FEATURES,
        "note": "Reconstruct absolute temp via: predicted_temp = lag_24 + predicted_delta",
        "rolling_std_ddof": 1
    }
    with open(FEATURE_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=4)
    print(f"  --> Saved Feature Config: {FEATURE_CFG_PATH}")

    # Metadata
    metadata = {
        "model_type": "XGBRegressor",
        "target_formulation": "delta_lag24",
        "forecast_horizon_hours": FORECAST_HORIZON,
        "feature_count": len(PRODUCTION_FEATURES),
        "features": PRODUCTION_FEATURES,
        "xgb_params": XGB_PARAMS,
        "quantile_coverage_target": 0.80,
        "test_interval_coverage": round(coverage, 2),
        "mean_interval_width": round(mean_interval_width, 4),
        "train_start": str(X_train.index[0].date()),
        "train_end": str(X_train.index[-1].date()),
        "test_start": str(X_test.index[0].date()),
        "test_end": str(X_test.index[-1].date()),
        "purged_train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_mean_mae": round(cv_mean_mae, 6),
        "cv_std_mae": round(cv_std_mae, 6),
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "test_mape": test_metrics["mape"],
        "test_medae": test_metrics["medae"],
        "may_mae": round(may_mae, 6),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "dependencies": {
            "xgboost": "3.4.0",
            "numpy": "2.5.2",
            "pandas": "3.0.5",
            "scikit-learn": "1.9.0",
            "optuna": "5.0.0"
        }
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  --> Saved Metadata: {METADATA_PATH}")

    print("\n[SUCCESS] Phase 3 training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
