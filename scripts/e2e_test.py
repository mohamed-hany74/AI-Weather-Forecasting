"""
E2E Test Script: Validates full web workflow and API contract.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

import django
django.setup()

from django.test import Client


def run_e2e_test():
    print("=" * 70)
    print("      WeatherCast AI — End-to-End System Test")
    print("=" * 70)

    client = Client()

    # 1. Dashboard View
    res_dash = client.get('/')
    assert res_dash.status_code == 200, "Dashboard root failed to render"
    assert b"WeatherCast" in res_dash.content, "Brand missing in dashboard"
    print("1. Dashboard HTML Template Rendering -> 200 OK")

    # 2. Health Endpoint
    res_health = client.get('/api/health/')
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"
    print("2. Health Endpoint (/api/health/) -> 200 OK")

    # 3. Model Info Endpoint
    res_info = client.get('/api/model-info/')
    assert res_info.status_code == 200
    assert res_info.json()["model"]["feature_count"] == 15
    print("3. Model Info Endpoint (/api/model-info/) -> 200 OK (15 features)")

    # 4. Predict Endpoint
    sample_history = [28.0] * 168
    payload = {
        "current_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z"),
        "latitude": 30.0444,
        "longitude": 31.2357,
        "temperature_history_168h": sample_history
    }
    res_pred = client.post('/api/predict/', data=json.dumps(payload), content_type='application/json')
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert data["status"] == "success"
    pred_val = data["prediction"]["predicted_temperature_2m"]
    print(f"4. Prediction Endpoint (/api/predict/) -> 200 OK (Predicted: {pred_val:.2f}°C)")

    print("\n[SUCCESS] All End-to-End System Tests Passed.")


if __name__ == "__main__":
    run_e2e_test()
