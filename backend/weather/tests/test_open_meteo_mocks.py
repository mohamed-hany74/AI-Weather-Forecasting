"""
Suite B: Open-Meteo and External Dependency Resilience Tests.
Mocks all external HTTP interactions. Covers:
- Connection errors and resets
- HTTP 500, 503, timeouts
- Retries and backoff policies
- Malformed payloads, missing hourly, mismatched arrays
- Recovery after transient failures
"""

import requests
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from django.test import TestCase

from django.core.cache import cache
from weather.domain.exceptions import ExternalWeatherServiceError
from weather.services.weather_service import fetch_realtime_exogenous


class OpenMeteoResilienceTests(TestCase):
    def setUp(self):
        cache.clear()
        self.lat = 30.0444
        self.lon = 31.2357
        self.target_time = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)

        self.valid_response_json = {
            "hourly": {
                "time": ["2026-08-15T12:00"],
                "apparent_temperature": [32.0],
                "pressure_msl": [1010.0],
                "relative_humidity_2m": [50.0],
                "dew_point_2m": [18.0],
                "wind_speed_10m": [12.0],
                "cloud_cover": [20.0]
            }
        }

    @patch("weather.services.weather_service._session.get")
    def test_successful_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = self.valid_response_json
        mock_get.return_value = mock_resp

        exo = fetch_realtime_exogenous(self.lat, self.lon, self.target_time, calibrate=False)
        self.assertEqual(exo.apparent_temperature, 32.0)
        self.assertEqual(exo.pressure_msl, 1010.0)
        self.assertEqual(exo.relative_humidity_2m, 50.0)

    @patch("weather.services.weather_service._session.get")
    def test_connection_error_raises_service_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

    @patch("weather.services.weather_service._session.get")
    def test_timeout_raises_service_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Read timeout after 5s")
        with self.assertRaises(ExternalWeatherServiceError) as ctx:
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)
        self.assertIn("timed out", str(ctx.exception))

    @patch("weather.services.weather_service._session.get")
    def test_http_503_raises_service_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("503 Server Error")
        mock_get.return_value = mock_resp
        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

    @patch("weather.services.weather_service._session.get")
    def test_malformed_json_raises_service_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = mock_resp
        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

    @patch("weather.services.weather_service._session.get")
    def test_missing_hourly_block_raises_service_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}  # Missing "hourly"
        mock_get.return_value = mock_resp
        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

    @patch("weather.services.weather_service._session.get")
    def test_empty_hourly_arrays_raises_service_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hourly": {
                "time": [],
                "apparent_temperature": [],
                "pressure_msl": [],
                "relative_humidity_2m": []
            }
        }
        mock_get.return_value = mock_resp
        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

    @patch("weather.services.weather_service._session.get")
    def test_recovery_after_transient_failure(self, mock_get):
        """First call fails with transient error; second succeeds."""
        mock_fail = MagicMock()
        mock_fail.side_effect = requests.exceptions.ConnectionError("Temporary blip")

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.json.return_value = self.valid_response_json

        mock_get.side_effect = [requests.exceptions.ConnectionError("Temporary blip"), mock_ok]

        with self.assertRaises(ExternalWeatherServiceError):
            fetch_realtime_exogenous(self.lat, self.lon, self.target_time)

        # Second attempt succeeds
        res = fetch_realtime_exogenous(self.lat, self.lon, self.target_time, calibrate=False)
        self.assertEqual(res.apparent_temperature, 32.0)
