"""
Unit tests for Feature Service and mathematical parity.
"""

import numpy as np
from datetime import datetime
from django.test import TestCase

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features


class FeatureServiceTests(TestCase):
    def setUp(self):
        self.exogenous = ExogenousWeather(
            apparent_temperature=32.5,
            pressure_msl=1011.2,
            relative_humidity_2m=58.0
        )

    def test_feature_vector_count_and_order(self):
        """Feature vector must contain exactly 15 features matching contract order."""
        history = [25.0] * 168
        dt = datetime(2024, 6, 15, 12, 0, 0)
        vector = build_production_features(history, dt, self.exogenous)

        self.assertEqual(len(vector), REQUIRED_FEATURE_COUNT)
        self.assertEqual(vector[0], 32.5)  # apparent_temperature
        self.assertEqual(vector[1], 1011.2) # pressure_msl
        self.assertEqual(vector[2], 58.0)  # relative_humidity_2m

    def test_synthetic_lag_sequence_alignment(self):
        """
        Synthetic test: history = [0, 1, 2, ..., 167].
        Verifies:
          lag_1   = 167 (history[-1])
          lag_24  = 144 (history[-24])
          lag_72  = 96  (history[-72])
          lag_168 = 0   (history[0])
          rolling_max_6   = 167 (max(162..167))
          rolling_max_24  = 167 (max(144..167))
          rolling_mean_24 = mean(144..167)
          rolling_std_24  = sample std(144..167, ddof=1)
        """
        history = [float(i) for i in range(168)]
        dt = datetime(2024, 6, 15, 12, 0, 0)
        vector = build_production_features(history, dt, self.exogenous)
        feat = dict(zip(FEATURE_NAMES, vector))

        self.assertEqual(feat["temperature_2m_lag_1"], 167.0)
        self.assertEqual(feat["temperature_2m_lag_24"], 144.0)
        self.assertEqual(feat["temperature_2m_lag_72"], 96.0)
        self.assertEqual(feat["temperature_2m_lag_168"], 0.0)
        self.assertEqual(feat["temperature_2m_rolling_max_6"], 167.0)
        self.assertEqual(feat["temperature_2m_rolling_max_24"], 167.0)

        exp_mean = float(np.mean(range(144, 168)))
        exp_std = float(np.std(range(144, 168), ddof=1))
        self.assertAlmostEqual(feat["temperature_2m_rolling_mean_24"], exp_mean)
        self.assertAlmostEqual(feat["temperature_2m_rolling_std_24"], exp_std)

    def test_cyclical_leap_year_equations(self):
        """Cyclical formulas for 2024 (leap year: 366 days)."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        history = [20.0] * 168
        vector = build_production_features(history, dt, self.exogenous)
        feat = dict(zip(FEATURE_NAMES, vector))

        # hour=0 -> cos(0) = 1.0
        self.assertAlmostEqual(feat["hour_cos"], 1.0)
        # month=1 -> cos(2*pi*(1-1)/12) = 1.0
        self.assertAlmostEqual(feat["month_cos"], 1.0)
        # day 1 in 366 -> sin(0) = 0.0, cos(0) = 1.0
        self.assertAlmostEqual(feat["dayofyear_sin"], 0.0)
        self.assertAlmostEqual(feat["dayofyear_cos"], 1.0)
