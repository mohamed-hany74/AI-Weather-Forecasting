"""
Unit tests for Model Service.
"""

from django.test import TestCase
from weather.services.model_service import predict_temperature, get_safe_model_info, load_model_artifacts
from weather.domain.exceptions import ModelInferenceError


class ModelServiceTests(TestCase):
    def setUp(self):
        load_model_artifacts()

    def test_model_info_schema(self):
        """get_safe_model_info returns expected keys and 15 features."""
        info = get_safe_model_info()
        self.assertIn("model_type", info)
        self.assertEqual(info["feature_count"], 15)
        self.assertEqual(info["forecast_horizon_hours"], 24)
        self.assertIn("metrics", info)

    def test_valid_inference(self):
        """Model produces finite numeric prediction on realistic Cairo vector."""
        vec = [
            32.0, 1010.0, 50.0,  # apparent_temp, pressure, humidity
            0.0, -0.866, -0.68, -0.73,  # hour_cos, month_cos, doy_sin, doy_cos
            28.0, 27.5, 27.0, 26.5,  # lags 1, 24, 72, 168
            34.0, 34.0, 28.5, 4.0   # roll max 6, max 24, mean 24, std 24
        ]
        pred = predict_temperature(vec)
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 15.0)
        self.assertLess(pred, 45.0)

    def test_invalid_feature_length_raises_error(self):
        """Invalid feature length must raise ModelInferenceError."""
        with self.assertRaises(ModelInferenceError):
            predict_temperature([1.0] * 10)
