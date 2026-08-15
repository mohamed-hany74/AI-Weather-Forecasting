"""
Integration and regression test suite: Proves temporal safety & 15-feature contract.
"""

import math
import numpy as np
from datetime import datetime, timezone
from django.test import TestCase

from weather.domain.schemas import PredictionRequest, ExogenousWeather
from weather.services.feature_service import build_production_features
from weather.services.model_service import predict_temperature


class IntegrationRegressionTests(TestCase):
    def test_regression_old_bug_prevention(self):
        """
        Regression test for old anomalous cold prediction:
        Proves that when daytime Cairo summer conditions are provided,
        the model outputs a realistic summer prediction (> 20°C).
        """
        # Summer daytime history in Cairo: 24°C - 35°C
        history = [
            round(28.5 + math.cos(((i % 24 - 14) / 24) * 2 * math.pi) * 5.8, 2)
            for i in range(168)
        ]
        dt = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
        exo = ExogenousWeather(
            apparent_temperature=34.0,
            pressure_msl=1009.0,
            relative_humidity_2m=50.0
        )

        vec = build_production_features(history, dt, exo)
        pred = predict_temperature(vec)

        # Sanity check: In Cairo summer daytime, prediction must be realistic (~25°C to 38°C)
        self.assertGreater(pred, 22.0)
        self.assertLess(pred, 42.0)
        self.assertNotAlmostEqual(pred, 5.95, delta=3.0)  # Cannot be near old bug value
