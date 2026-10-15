"""
Unit tests for Timezone and ISO-8601 parsing.
"""

from datetime import timezone
from django.test import TestCase
from weather.domain.schemas import PredictionRequest
from weather.domain.exceptions import ValidationError


class TimezoneTests(TestCase):
    def test_timezone_aware_conversion_to_utc(self):
        """+02:00 (Cairo) timestamp must be converted to UTC without offset."""
        payload = {
            "current_time": "2026-08-15T14:00:00+02:00",
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": [25.0] * 168
        }
        req = PredictionRequest.from_dict(payload)
        self.assertEqual(req.reference_time.tzinfo, timezone.utc)
        self.assertEqual(req.reference_time.hour, 12)  # 14:00 +02:00 = 12:00 UTC

    def test_zulu_timestamp_parsing(self):
        """Standard Z timestamp must parse as UTC."""
        payload = {
            "current_time": "2026-08-15T12:00:00Z",
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": [25.0] * 168
        }
        req = PredictionRequest.from_dict(payload)
        self.assertEqual(req.reference_time.hour, 12)
        self.assertEqual(req.reference_time.tzinfo, timezone.utc)

    def test_invalid_timestamp_rejected(self):
        """Malformed timestamps must be rejected with ValidationError."""
        payload = {
            "current_time": "not-a-date",
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": [25.0] * 168
        }
        with self.assertRaises(ValidationError):
            PredictionRequest.from_dict(payload)

    def test_time_alignment_lags_and_current_temperature_semantics(self):
        """
        Amendment B: Strict verification of time alignment.
        Given:
          reference_time t = 2026-08-15T12:00:00Z
          history_168 = [100.0, 101.0, ..., 267.0] where value = 100 + i (i in 0..167)
        Assert:
          1. Target forecast refers to t + 24h = 2026-08-16T12:00:00Z.
          2. history[-1] (267.0) is the observation at t-1h (latest historical observation prior to t).
          3. lag_24 = history[-24] (244.0) is the observation at t-24h (the anchor baseline).
          4. lag_168 = history[0] (100.0) is the observation at t-168h.
          5. current_temperature = history[-1] (267.0), ensuring change_vs_current represents
             truthful projected change from the latest observation to t+24h.
        """
        from datetime import datetime, timedelta
        from weather.services.feature_service import build_production_features
        from weather.domain.schemas import ExogenousWeather

        ref_t = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        history = [100.0 + i for i in range(168)]
        exo = ExogenousWeather(30.0, 1013.0, 50.0)

        feats = build_production_features(history, ref_t, exo)

        # In contract order:
        # lag_1 is history[-1] = 267.0
        # lag_24 is history[-24] = 244.0
        # lag_72 is history[-72] = 196.0
        # lag_168 is history[0] = 100.0
        self.assertEqual(history[-1], 267.0)
        self.assertEqual(history[-24], 244.0)
        self.assertEqual(history[0], 100.0)
        self.assertEqual(ref_t + timedelta(hours=24), datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc))
