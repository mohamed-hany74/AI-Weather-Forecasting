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
        self.assertEqual(data["model"]["feature_count"], 21)


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
        self.assertNotIn("confidence_interval_80", data["prediction"])
        expected_current = self.valid_payload["temperature_history_168h"][-1]
        self.assertEqual(data["prediction"]["current_temperature"], round(expected_current, 2))
        self.assertEqual(
            data["prediction"]["change_vs_current"],
            round(data["prediction"]["predicted_temperature_2m"] - expected_current, 2)
        )
        self.assertEqual(data["prediction"]["typical_error_celsius"], 1.4)
        self.assertIn("Historical average error", data["prediction"]["error_reference_note"])

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

    def test_readiness_probe_returns_200(self):
        """GET /api/ready/ returns 200 when model is initialized."""
        res = self.client.get('/api/ready/')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ready")

    @patch("weather.api.views.is_model_ready")
    def test_readiness_probe_returns_503_when_not_ready(self, mock_ready):
        """GET /api/ready/ returns 503 when model is not ready."""
        mock_ready.return_value = False
        res = self.client.get('/api/ready/')
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["code"], "NOT_READY")

    def test_exogenous_out_of_bounds_validation_returns_400(self):
        """Supplied exogenous weather with out-of-bounds values returns 400."""
        # 1. Invalid humidity (> 100%)
        payload_bad_hum = dict(self.valid_payload)
        payload_bad_hum["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 1013.0,
            "relative_humidity_2m": 120.0
        }
        res = self.client.post('/api/predict/', data=json.dumps(payload_bad_hum), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

        # 2. Invalid pressure (< 870 hPa)
        payload_bad_press = dict(self.valid_payload)
        payload_bad_press["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 750.0,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post('/api/predict/', data=json.dumps(payload_bad_press), content_type="application/json")
        self.assertEqual(res.status_code, 400)

        # 3. Non-finite value (NaN / Inf)
        payload_bad_nan = dict(self.valid_payload)
        payload_bad_nan["exogenous_weather"] = {
            "apparent_temperature": "NaN",
            "pressure_msl": 1013.0,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post('/api/predict/', data=json.dumps(payload_bad_nan), content_type="application/json")
        self.assertEqual(res.status_code, 400)

    @patch("weather.services.forecast_service.predict_temperature")
    @patch("weather.services.forecast_service.fetch_realtime_exogenous")
    def test_model_non_finite_output_returns_500(self, mock_fetch, mock_predict):
        """Model producing non-finite output results in HTTP 500 error."""
        from weather.domain.exceptions import ModelInferenceError
        mock_fetch.return_value = ExogenousWeather(30.0, 1010.0, 50.0)
        mock_predict.side_effect = ModelInferenceError("Non-finite model prediction generated (nan).")

        res = self.client.post('/api/predict/', data=json.dumps(self.valid_payload), content_type="application/json")
        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.json()["code"], "MODEL_INFERENCE_ERROR")

    @patch("weather.services.forecast_service.fetch_realtime_exogenous")
    def test_rate_limiting_returns_429(self, mock_fetch):
        """Surpassing 60 requests per minute triggers 429 Too Many Requests."""
        from django.core.cache import cache
        cache.clear()
        mock_fetch.return_value = ExogenousWeather(30.0, 1010.0, 50.0)

        # Make 60 successful requests
        for _ in range(60):
            res = self.client.post('/api/predict/', data=json.dumps(self.valid_payload), content_type="application/json")
            self.assertEqual(res.status_code, 200)

        # 61st request must be throttled with 429
        throttled_res = self.client.post('/api/predict/', data=json.dumps(self.valid_payload), content_type="application/json")
        self.assertEqual(throttled_res.status_code, 429)
        self.assertEqual(throttled_res.json()["code"], "RATE_LIMIT_EXCEEDED")
        self.assertIn("Retry-After", throttled_res)

    @patch("weather.services.forecast_service.fetch_168h_history_and_exogenous")
    def test_forecast_easy_endpoint_success(self, mock_bundle):
        """POST /api/forecast/ returns 200 with minimal or empty payload using real Open-Meteo bundle."""
        from datetime import datetime, timezone
        from weather.domain.schemas import ExogenousWeather

        ref_time = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        mock_bundle.return_value = {
            "history_168": [26.0] * 168,
            "current_temperature": 27.5,
            "exogenous": ExogenousWeather(
                apparent_temperature=32.0,
                pressure_msl=1011.0,
                relative_humidity_2m=45.0,
                dew_point_2m=14.0,
                wind_speed_10m=12.0,
                cloud_cover=15.0,
                pressure_change_3h=-0.5,
                pressure_change_6h=-1.0
            ),
            "reference_time": ref_time
        }

        # Request with default coordinates
        res = self.client.post(
            '/api/forecast/',
            data=json.dumps({"latitude": 30.0444, "longitude": 31.2357, "current_time": "2026-08-15T12:00:00Z"}),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["prediction"]["target_variable"], "temperature_2m")
        self.assertEqual(data["prediction"]["current_temperature"], 27.5)
        self.assertIsInstance(data["prediction"]["predicted_temperature_2m"], float)
        self.assertNotIn("confidence_interval_80", data["prediction"])
        self.assertEqual(data["prediction"]["typical_error_celsius"], 1.4)
        self.assertEqual(
            data["prediction"]["change_vs_current"],
            round(data["prediction"]["predicted_temperature_2m"] - 27.5, 2)
        )

    def test_forecast_easy_invalid_coords(self):
        """POST /api/forecast/ with invalid coordinates returns 400."""
        res = self.client.post(
            '/api/forecast/',
            data=json.dumps({"latitude": 999.0, "longitude": 31.2357}),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    @patch("weather.services.forecast_service.fetch_168h_history_and_exogenous")
    def test_forecast_easy_open_meteo_down_returns_503(self, mock_bundle):
        """POST /api/forecast/ returns 503 when Open-Meteo fails."""
        mock_bundle.side_effect = ExternalWeatherServiceError("Open-Meteo connection refused")
        res = self.client.post('/api/forecast/', data=json.dumps({}), content_type="application/json")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["code"], "EXTERNAL_SERVICE_ERROR")

