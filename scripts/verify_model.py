"""
Verification Script: Model Serialization & Performance.
"""

import os
import sys
import xgboost as xgb
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from weather.services.model_service import load_model_artifacts, predict_temperature, get_safe_model_info


def verify_model():
    print("=" * 70)
    print("       MODEL SERIALIZATION & INFERENCE VERIFICATION")
    print("=" * 70)

    load_model_artifacts()
    info = get_safe_model_info()

    print(f"Model Type    : {info['model_type']}")
    print(f"Feature Count : {info['feature_count']}")
    print(f"Test MAE      : {info['metrics']['test_mae']}°C")
    print(f"Test RMSE     : {info['metrics']['test_rmse']}°C")
    print(f"Test R²       : {info['metrics']['test_r2']}")

    # Run sample inference
    sample_vec = [
        32.0, 1010.0, 55.0,
        0.0, -0.866, -0.68, -0.73,
        28.0, 27.5, 27.0, 26.5,
        34.0, 34.0, 28.5, 4.0
    ]

    pred = predict_temperature(sample_vec)
    print(f"\nSample Prediction for Cairo summer day: {pred:.2f}°C")
    assert 20.0 <= pred <= 45.0, f"Sample prediction {pred}°C out of bounds!"
    print("[PASS] Model serialization and inference verified.")


if __name__ == "__main__":
    verify_model()
