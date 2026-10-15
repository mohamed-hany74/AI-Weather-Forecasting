"""
Suite F: Feature Properties and Time-Series Robustness Tests.
Validates mathematical and behavioral invariants of feature engineering:
- compute_time_series_features with arbitrary/noisy sequences
- hours_since_daily_max bounded in [1.0, 24.0]
- non-negative rolling standard deviation
- determinism on identical inputs
- strict length validation (raises error if len != 168)
"""

import math
import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
from hypothesis.extra.django import TestCase
from hypothesis import given, settings, strategies as st

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features
from ml.training.features import compute_time_series_features


class FeaturePropertiesRobustnessTests(TestCase):
    def setUp(self):
        self.exogenous = ExogenousWeather(
            apparent_temperature=30.0,
            pressure_msl=1010.0,
            relative_humidity_2m=50.0,
            dew_point_2m=15.0,
            wind_speed_10m=12.0,
            cloud_cover=25.0,
            pressure_change_3h=0.5,
            pressure_change_6h=1.0
        )
        self.ref_time = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_hours_since_daily_max_bounds_and_correctness(self):
        """hours_since_daily_max must always be finite and within [1.0, 24.0]."""
        # Case 1: Max at t-1h (last element of history) -> hours_since_max = 1.0
        hist_1 = [20.0] * 167 + [35.0]
        feats_1 = compute_time_series_features(hist_1)
        self.assertEqual(feats_1["hours_since_daily_max"], 1.0)

        # Case 2: Max at t-24h (24th from end) -> hours_since_max = 24.0
        hist_24 = [20.0] * 144 + [35.0] + [20.0] * 23
        feats_24 = compute_time_series_features(hist_24)
        self.assertEqual(feats_24["hours_since_daily_max"], 24.0)

        # Case 3: Constant temperatures across 24h -> returns valid finite number in [1, 24]
        hist_const = [25.0] * 168
        feats_const = compute_time_series_features(hist_const)
        self.assertGreaterEqual(feats_const["hours_since_daily_max"], 1.0)
        self.assertLessEqual(feats_const["hours_since_daily_max"], 24.0)

    def test_rolling_std_is_non_negative(self):
        """Standard deviation must be >= 0.0 for any finite sequence."""
        hist = [float(i % 10) for i in range(168)]
        feats = compute_time_series_features(hist)
        self.assertGreaterEqual(feats["temperature_2m_rolling_std_24"], 0.0)

        hist_const = [18.0] * 168
        feats_const = compute_time_series_features(hist_const)
        self.assertAlmostEqual(feats_const["temperature_2m_rolling_std_24"], 0.0)

    def test_strict_history_length_enforcement(self):
        """Sequences not equal to 168 items must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_time_series_features([20.0] * 100)
        with self.assertRaises(ValueError):
            compute_time_series_features([20.0] * 200)

    def test_determinism_on_identical_inputs(self):
        """Calling build_production_features twice with same input yields identical results."""
        hist = [22.0 + (i % 7) for i in range(168)]
        vec1 = build_production_features(hist, self.ref_time, self.exogenous)
        vec2 = build_production_features(hist, self.ref_time, self.exogenous)
        self.assertEqual(vec1, vec2)
        self.assertEqual(len(vec1), REQUIRED_FEATURE_COUNT)

    @given(
        hist=st.lists(
            st.floats(min_value=-40.0, max_value=60.0, allow_nan=False, allow_infinity=False),
            min_size=168,
            max_size=168
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_hypothesis_features_always_finite_and_bounded(self, hist):
        """Property: for any 168-element sequence, all engineered features are finite."""
        feats = compute_time_series_features(hist)
        for k, v in feats.items():
            self.assertTrue(math.isfinite(v), f"Feature {k} became non-finite: {v}")
        self.assertGreaterEqual(feats["hours_since_daily_max"], 1.0)
        self.assertLessEqual(feats["hours_since_daily_max"], 24.0)
        self.assertGreaterEqual(feats["temperature_2m_rolling_std_24"], 0.0)
