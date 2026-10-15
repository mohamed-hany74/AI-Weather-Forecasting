"""
Suite C: Machine Learning Regression Gate.
Evaluates the existing trained model on a fixed, reproducible, genuinely held-out
chronological test dataset (2018-2019, 17,505 rows).
Enforces the mandatory gate: MAE <= 1.40 °C without retraining.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from django.test import TestCase
from django.conf import settings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from ml.training.features import engineer_dataframe_features


class MLRegressionGateTests(TestCase):
    def test_ml_generalization_gate_held_out_mae(self):
        """
        MANDATORY ML GATE:
        Evaluates existing model artifact on out-of-sample held-out test split.
        Asserts MAE <= 1.40 °C.
        """
        # 1. Locate dataset and artifacts
        data_path = os.path.join(settings.BASE_DIR.parent, "ml", "data", "weather.csv")
        model_path = os.path.join(settings.BASE_DIR.parent, "ml", "models", "xgboost_weather_model.json")
        meta_path = os.path.join(settings.BASE_DIR.parent, "ml", "models", "model_metadata.json")

        self.assertTrue(os.path.exists(data_path), f"Data missing: {data_path}")
        self.assertTrue(os.path.exists(model_path), f"Model missing: {model_path}")
        self.assertTrue(os.path.exists(meta_path), f"Metadata missing: {meta_path}")

        # 2. Load dataset and generate exact features
        df_raw = pd.read_csv(data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
        df_raw.set_index("date", inplace=True)
        df_raw.sort_index(inplace=True)
        df = df_raw.asfreq("h").ffill().bfill()

        df_feat = engineer_dataframe_features(df)
        df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target", "target_abs"]).copy()

        # 3. Canonical 80/20 chronological split (test period: 2017-12-31 to 2019-12-30)
        n_total = len(df)
        n_test = int(n_total * 0.20)
        split_dt = df.index[n_total - n_test]

        df_test = df_clean[df_clean.index >= split_dt].copy()
        self.assertEqual(len(df_test), 17505, "Test row count must be exactly 17,505 rows")

        X_test = df_test[FEATURE_NAMES]
        lag24_test = df_test["temperature_2m_lag_24"].values
        y_test_abs = df_test["target_abs"].values

        # 4. Load trained model without retraining
        booster = xgb.Booster()
        booster.load_model(model_path)

        dmat = xgb.DMatrix(X_test, feature_names=FEATURE_NAMES)
        pred_deltas = booster.predict(dmat)
        pred_abs = lag24_test + pred_deltas

        # 5. Calculate Metrics
        mae = float(mean_absolute_error(y_test_abs, pred_abs))
        rmse = float(np.sqrt(mean_squared_error(y_test_abs, pred_abs)))
        r2 = float(r2_score(y_test_abs, pred_abs))

        # Check against metadata recorded values
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertAlmostEqual(mae, metadata["test_mae"], places=4, msg="Evaluation MAE must match metadata")

        # 6. ENFORCE MANDATORY QUALITY GATE
        # Gate: MAE <= 1.40 °C
        self.assertLessEqual(
            mae,
            1.40,
            f"FAIL: Generalization MAE {mae:.4f}°C exceeds mandatory quality gate threshold of 1.40°C"
        )
        self.assertGreater(r2, 0.90, f"R² {r2:.4f} must exceed 0.90")
