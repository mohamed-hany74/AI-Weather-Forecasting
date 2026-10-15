"""
Suite A: Input Contracts, Boundaries, and Fuzzing Tests.
Validates strict schema compliance, boundary condition rejection (HTTP 400),
non-finite handling (NaN, Inf), malformed JSON, and property-based fuzzing.
"""

import json
import math
from hypothesis import given, settings, strategies as st
from hypothesis.extra.django import TestCase
from django.test import Client

from django.core.cache import cache

from weather.domain.schemas import ExogenousWeather


class InputContractsBoundaryTests(TestCase):

    def setUp(self):
        cache.clear()
        self.client = Client()
        self.valid_payload = {
            "current_time": "2026-08-15T12:00:00Z",
            "latitude": 30.0444,
            "longitude": 31.2357,
            "temperature_history_168h": [25.0 + (i % 8) * 0.5 for i in range(168)]
        }


    # =========================================================================
    # A1. Boundary Violations on Exogenous Inputs
    # =========================================================================
    def test_relative_humidity_upper_boundary_violation(self):
        """relative_humidity_2m > 100.0 (100.0001) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 1013.0,
            "relative_humidity_2m": 100.0001
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    def test_relative_humidity_lower_boundary_violation(self):
        """relative_humidity_2m < 0.0 (-0.0001) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 1013.0,
            "relative_humidity_2m": -0.0001
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    def test_pressure_msl_lower_boundary_violation(self):
        """pressure_msl < 870.0 (869.99) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 869.99,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    def test_pressure_msl_upper_boundary_violation(self):
        """pressure_msl > 1085.0 (1085.01) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": 25.0,
            "pressure_msl": 1085.01,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    def test_apparent_temperature_lower_boundary_violation(self):
        """apparent_temperature < -60.0 (-60.01) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": -60.01,
            "pressure_msl": 1013.0,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    def test_apparent_temperature_upper_boundary_violation(self):
        """apparent_temperature > 60.0 (60.01) must return HTTP 400."""
        payload = dict(self.valid_payload)
        payload["exogenous_weather"] = {
            "apparent_temperature": 60.01,
            "pressure_msl": 1013.0,
            "relative_humidity_2m": 50.0
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "VALIDATION_ERROR")

    # =========================================================================
    # A2. Non-Finite Values (NaN, Infinity, -Infinity)
    # =========================================================================
    def test_nan_in_history_returns_400(self):
        """NaN inside temperature_history_168h returns HTTP 400."""
        payload = dict(self.valid_payload)
        history = list(payload["temperature_history_168h"])
        history[50] = float("nan")
        # Direct string replacement for valid JSON
        body = json.dumps(payload).replace(f"{payload['temperature_history_168h'][50]}", "NaN", 1)
        res = self.client.post("/api/predict/", data=body, content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_infinity_in_coordinates_returns_400(self):
        """Infinity latitude returns HTTP 400."""
        body = '{"current_time":"2026-08-15T12:00:00Z","latitude":Infinity,"longitude":31.2357,"temperature_history_168h":[25.0]}'
        res = self.client.post("/api/predict/", data=body, content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_null_value_in_required_field_returns_400(self):
        """Null for current_time returns HTTP 400."""
        payload = dict(self.valid_payload)
        payload["current_time"] = None
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)

    # =========================================================================
    # A3. Missing and Malformed Inputs
    # =========================================================================
    def test_omitted_latitude_returns_400(self):
        payload = dict(self.valid_payload)
        del payload["latitude"]
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_empty_json_object_returns_400(self):
        res = self.client.post("/api/predict/", data="{}", content_type="application/json")
        self.assertEqual(res.status_code, 400)

    def test_empty_request_body_returns_400(self):
        res = self.client.post("/api/predict/", data="", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "EMPTY_BODY")

    def test_malformed_json_syntax_returns_400(self):
        res = self.client.post("/api/predict/", data="{invalid json::", content_type="application/json")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["code"], "JSON_PARSE_ERROR")

    def test_incorrect_http_method_returns_405(self):
        res = self.client.get("/api/predict/")
        self.assertEqual(res.status_code, 405)

    def test_nested_object_where_scalar_expected(self):
        payload = dict(self.valid_payload)
        payload["latitude"] = {"nested": 30.0}
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)

    # =========================================================================
    # A4. Property-Based Fuzzing with Hypothesis
    # =========================================================================
    @given(
        lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False),
        lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False),
        temps=st.lists(
            st.floats(min_value=-15.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=168,
            max_size=168
        )
    )
    @settings(max_examples=25, deadline=None)
    def test_hypothesis_fuzz_valid_requests(self, lat, lon, temps):
        """Property: any valid atmospheric parameters within contract never crash unhandled."""
        payload = {
            "current_time": "2026-08-15T12:00:00Z",
            "latitude": lat,
            "longitude": lon,
            "temperature_history_168h": temps,
            "exogenous_weather": {
                "apparent_temperature": 25.0,
                "pressure_msl": 1013.0,
                "relative_humidity_2m": 50.0,
                "dew_point_2m": min(10.0, temps[-1] + 0.5),
                "wind_speed_10m": 12.0,
                "cloud_cover": 20.0,
                "pressure_change_3h": 0.5,
                "pressure_change_6h": 1.0
            }
        }
        res = self.client.post("/api/predict/", data=json.dumps(payload), content_type="application/json")
        # Must be either 200 (success) or 429 (rate-limited) - NEVER 500
        self.assertIn(res.status_code, [200, 429])
        if res.status_code == 200:
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertTrue(math.isfinite(data["prediction"]["predicted_temperature_2m"]))
            self.assertTrue(math.isfinite(data["prediction"]["current_temperature"]))
            self.assertTrue(math.isfinite(data["prediction"]["change_vs_current"]))
