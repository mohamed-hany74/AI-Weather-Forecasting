"""
Reproducible Training Pipeline for WeatherCast AI.
Trains XGBoost Regressor with 15 causal features, purged chronological splitting, and exports production JSON artifacts.
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
FEATURE_CFG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

TEST_SIZE = 0.20
RANDOM_STATE = 42
ROLLING_WIN = 720

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}


def run_training_pipeline():
    print("=" * 70)
    print("      WeatherCast AI — Training Pipeline Execution")
    print("=" * 70)

    # 1. Load Dataset
    print(f"\n[1/6] Loading dataset from: {DATA_PATH}")
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
    print("[2/6] Engineering 15 causal features...")
    df_clean = pd.concat([df_train_raw, df_test_raw])
    df_feat = engineer_dataframe_features(df_clean)
    df_valid = df_feat.dropna(subset=PRODUCTION_FEATURES + ["target"]).copy()

    # 5. Purge & Embargo Boundary Enforcement
    print("[3/6] Enforcing 24h purge & embargo boundary...")
    purge_cutoff = split_timestamp - pd.Timedelta(hours=FORECAST_HORIZON)
    df_tr = df_valid[df_valid.index < purge_cutoff].copy()
    df_te = df_valid[df_valid.index >= split_timestamp].copy()

    X_train = df_tr[PRODUCTION_FEATURES]
    y_train = df_tr["target"]
    X_test = df_te[PRODUCTION_FEATURES]
    y_test = df_te["target"]

    validate_temporal_boundary(df_tr.index[-1], df_te.index[0], FORECAST_HORIZON)
    validate_feature_matrix(X_train, PRODUCTION_FEATURES)
    validate_feature_matrix(X_test, PRODUCTION_FEATURES)
    validate_target_series(y_train)
    validate_target_series(y_test)

    print(f"  Train set: {len(X_train):,} rows ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"  Test set : {len(X_test):,} rows ({X_test.index[0].date()} to {X_test.index[-1].date()})")

    # 6. Walk-Forward Cross Validation
    print("\n[4/6] Running 5-Fold Walk-Forward Cross Validation on Purged Train Set...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        model_cv = XGBRegressor(**XGB_PARAMS)
        model_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds_val = model_cv.predict(X_train.iloc[val_idx])
        metrics_fold = evaluate_forecasts(y_train.iloc[val_idx].values, preds_val)
        cv_maes.append(metrics_fold["mae"])
        print(f"  Fold {fold}: MAE={metrics_fold['mae']:.4f}°C | RMSE={metrics_fold['rmse']:.4f}°C | R²={metrics_fold['r2']:.4f}")

    cv_mean_mae = float(np.mean(cv_maes))
    cv_std_mae = float(np.std(cv_maes))
    print(f"  CV Mean MAE: {cv_mean_mae:.4f}°C (±{cv_std_mae:.4f})")

    # 7. Final Model Training
    print(f"\n[5/6] Training Final XGBoost Production Model ({XGB_PARAMS['n_estimators']} trees)...")
    t0 = time.time()
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train)
    fit_duration = time.time() - t0
    print(f"  Trained in {fit_duration:.2f} seconds.")

    # 8. Unseen Test Evaluation
    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate_forecasts(y_test.values, y_test_pred)
    print("\n===============================================================")
    print("        UNSEEN OUT-OF-SAMPLE TEST EVALUATION RESULTS           ")
    print("===============================================================")
    for k, v in test_metrics.items():
        print(f"  {k.upper():<10}: {v}")
    print("===============================================================")

    # 9. Export Artifacts
    print(f"\n[6/6] Exporting production artifacts to: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Booster JSON
    final_model.get_booster().save_model(MODEL_OUT_PATH)
    print(f"  --> Saved Model JSON: {MODEL_OUT_PATH}")

    # Feature Config
    feature_config = {
        "target": TARGET,
        "forecast_horizon": FORECAST_HORIZON,
        "feature_count": len(PRODUCTION_FEATURES),
        "features": PRODUCTION_FEATURES,
        "note_exogenous": "apparent_temperature, pressure_msl, relative_humidity_2m are observed at t via Open-Meteo.",
        "rolling_std_ddof": 1
    }
    with open(FEATURE_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=4)
    print(f"  --> Saved Feature Config: {FEATURE_CFG_PATH}")

    # Metadata
    metadata = {
        "model_type": "XGBRegressor",
        "target": TARGET,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "feature_count": len(PRODUCTION_FEATURES),
        "features": PRODUCTION_FEATURES,
        "xgb_params": XGB_PARAMS,
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
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  --> Saved Metadata: {METADATA_PATH}")

    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
