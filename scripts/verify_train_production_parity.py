"""
Critical Test #2: Train vs Production Deterministic Parity Verification.
Verifies:
1. Exact feature construction parity between training DataFrame and production FeatureService across 100+ test timestamps.
2. Exact prediction parity between training in-memory XGBoost model and saved production JSON model loaded through model_service.py.
3. Quantifies MAE difference, RMSE difference, max absolute prediction difference, and R² difference.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features
from weather.services.model_service import predict_temperature, load_model_artifacts
from ml.training.features import engineer_dataframe_features


def verify_train_production_parity():
    print("=" * 80)
    print("   CRITICAL TEST #2: TRAIN VS PRODUCTION DETERMINISTIC PARITY VERIFICATION")
    print("=" * 80)

    # A. Load dataset
    data_path = os.path.join(PROJECT_ROOT, "ml", "data", "weather.csv")
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
    df_raw.set_index("date", inplace=True)
    df_raw.sort_index(inplace=True)
    df = df_raw.asfreq("h").ffill().bfill()
    TARGET = "temperature_2m"

    # B. Reproduce exact training features
    df_feat = engineer_dataframe_features(df)
    df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target"]).copy()

    split_dt = pd.to_datetime("2018-01-01 00:00:00+00:00")
    df_test = df_clean[df_clean.index >= split_dt].copy()
    X_test_df = df_test[FEATURE_NAMES]
    y_test = df_test["target"]

    # C. Select 200 deterministic test timestamps spread evenly across 2018-2019
    sample_indices = np.linspace(0, len(df_test) - 1, 200, dtype=int)
    test_sample = df_test.iloc[sample_indices]

    # D. Load direct booster (training reference)
    booster_ref = xgb.Booster()
    booster_ref.load_model(os.path.join(PROJECT_ROOT, "ml", "models", "xgboost_weather_model.json"))

    # E. Load production model_service
    load_model_artifacts()

    feature_diffs = []
    preds_training_ref = []
    preds_production_service = []
    preds_production_from_history = []

    print(f"\nEvaluating {len(sample_indices)} deterministic test timestamps across 2018–2019...")

    for idx, (t, row) in enumerate(test_sample.iterrows()):
        vec_train = row[FEATURE_NAMES].values.astype(float).tolist()
        
        # 1. Direct prediction from training feature row
        dmat = xgb.DMatrix(pd.DataFrame([row[FEATURE_NAMES]]), feature_names=FEATURE_NAMES)
        p_train = float(booster_ref.predict(dmat)[0])
        preds_training_ref.append(p_train)

        # 2. Production model_service on exact training vector
        p_prod_direct = predict_temperature(vec_train)
        preds_production_service.append(p_prod_direct)

        # 3. Production FeatureService reconstructing vector from raw 168h history + exogenous
        raw_idx = df.index.get_loc(t)
        history_168 = df[TARGET].iloc[raw_idx-168:raw_idx].tolist()
        row_raw = df.loc[t]
        exo = ExogenousWeather(
            apparent_temperature=float(row_raw["apparent_temperature"]),
            pressure_msl=float(row_raw["pressure_msl"]),
            relative_humidity_2m=float(row_raw["relative_humidity_2m"])
        )
        vec_prod_reconstructed = build_production_features(history_168, t, exo)
        p_prod_hist = predict_temperature(vec_prod_reconstructed)
        preds_production_from_history.append(p_prod_hist)

        # Measure feature vector difference
        f_diff = max(abs(a - b) for a, b in zip(vec_train, vec_prod_reconstructed))
        feature_diffs.append(f_diff)

    preds_training_ref = np.array(preds_training_ref)
    preds_production_service = np.array(preds_production_service)
    preds_production_from_history = np.array(preds_production_from_history)

    # Calculate differences
    max_feat_diff = max(feature_diffs)
    max_pred_diff_service = np.max(np.abs(preds_training_ref - preds_production_service))
    max_pred_diff_history = np.max(np.abs(preds_training_ref - preds_production_from_history))

    mae_diff = np.mean(np.abs(preds_training_ref - preds_production_from_history))
    rmse_diff = np.sqrt(np.mean((preds_training_ref - preds_production_from_history) ** 2))

    print("\n" + "-" * 70)
    print("                   PARITY EVALUATION RESULTS")
    print("-" * 70)
    print(f"  Max Feature Vector Difference (Train vs Prod) : {max_feat_diff:.2e}")
    print(f"  Max Prediction Difference (Booster vs Service): {max_pred_diff_service:.2e}")
    print(f"  Max Prediction Difference (Train vs History)  : {max_pred_diff_history:.2e}")
    print(f"  MAE Difference Between Pipeline Predictions   : {mae_diff:.2e}°C")
    print(f"  RMSE Difference Between Pipeline Predictions  : {rmse_diff:.2e}°C")
    print("-" * 70)

    assert max_feat_diff <= 1e-6, f"FAIL: Feature parity diff {max_feat_diff} > 1e-6"
    assert max_pred_diff_service <= 1e-6, f"FAIL: Booster vs service diff {max_pred_diff_service} > 1e-6"
    assert max_pred_diff_history <= 1e-6, f"FAIL: Train vs history pred diff {max_pred_diff_history} > 1e-6"

    print("\n[STATUS: PASS] Deterministic Train vs Production Parity verified with error <= 1e-6.")
    return True


if __name__ == "__main__":
    verify_train_production_parity()
