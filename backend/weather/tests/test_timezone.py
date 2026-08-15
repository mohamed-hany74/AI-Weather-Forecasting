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
