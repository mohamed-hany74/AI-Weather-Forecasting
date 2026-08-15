"""
Integration tests for API endpoints.
"""

import json
from unittest.mock import patch
from django.test import TestCase, Client

from weather.domain.schemas import ExogenousWeather
from weather.domain.exceptions import ExternalWeatherServiceError


class ApiEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.valid_payload = {
            "current_time": "2026-08-15T12:00:00Z",
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": [25.0 + (i % 8) * 0.5 for i in range(168)]
        }

    def test_health_check(self):
        """GET /api/health/ returns 200 OK."""
        res = self.client.get('/api/health/')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_model_info(self):
        """GET /api/model-info/ returns 200 OK with metadata."""
        res = self.client.get('/api/model-info/')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"]["feature_count"], 15)

    @patch("weather.services.forecast_service.fetch_realtime_exogenous")
    def test_predict_endpoint_success(self, mock_fetch):
        """POST /api/predict/ returns 200 and forecast."""
        mock_fetch.return_value = ExogenousWeather(
            apparent_temperature=33.0,
            pressure_msl=1010.0,
            relative_humidity_2m=55.0
        )
        res = self.client.post(
            '/api/predict/',
            data=json.dumps(self.valid_payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["prediction"]["forecast_time"], "2026-08-16T12:00:00Z")
        self.assertIsInstance(data["prediction"]["predicted_temperature_2m"], float)

    def test_predict_bad_json(self):
        """Invalid JSON returns 400."""
        res = self.client.post('/api/predict/', data="broken", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["status"], "error")

    def test_predict_invalid_history_length(self):
        """History < 168 returns 400."""
        bad_payload = dict(self.valid_payload)
        bad_payload["temperature_history_168h"] = [25.0] * 50
        res = self.client.post(
            '/api/predict/',
            data=json.dumps(bad_payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    @patch("weather.services.forecast_service.fetch_realtime_exogenous")
    def test_open_meteo_down_returns_503(self, mock_fetch):
        """Open-Meteo failure returns 503 Service Unavailable."""
        mock_fetch.side_effect = ExternalWeatherServiceError("Open-Meteo timeout")
        res = self.client.post(
            '/api/predict/',
            data=json.dumps(self.valid_payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["code"], "EXTERNAL_SERVICE_ERROR")
