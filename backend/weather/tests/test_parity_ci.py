"""
Suite D: Training-to-Production Feature Parity Automated Test.
Integrates verify_train_production_parity logic directly into Django's test runner.
Asserts that max absolute difference between offline DataFrame features and
production feature service is strictly < 1e-6 (target: 0.00e+00).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from django.test import TestCase
from django.conf import settings

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT
from weather.domain.schemas import ExogenousWeather
from weather.services.feature_service import build_production_features
from weather.services.model_service import predict_temperature, load_model_artifacts
from ml.training.features import engineer_dataframe_features


class TrainingProductionParityCITests(TestCase):
    def test_deterministic_training_production_parity_ci(self):
        """
        MANDATORY CI PARITY GATE:
        Compares offline training features with production feature service reconstruction
        across 100 deterministic evaluation timestamps.
        Asserts max_abs_diff <= 1e-6.
        """
        data_path = os.path.join(settings.BASE_DIR.parent, "ml", "data", "weather.csv")
        model_path = os.path.join(settings.BASE_DIR.parent, "ml", "models", "xgboost_weather_model.json")

        df_raw = pd.read_csv(data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
        df_raw.set_index("date", inplace=True)
        df_raw.sort_index(inplace=True)
        df = df_raw.asfreq("h").ffill().bfill()
        TARGET = "temperature_2m"

        df_feat = engineer_dataframe_features(df)
        df_clean = df_feat.dropna(subset=FEATURE_NAMES + ["target", "target_abs"]).copy()

        split_dt = pd.to_datetime("2018-01-01 00:00:00+00:00")
        df_test = df_clean[df_clean.index >= split_dt].copy()

        sample_indices = np.linspace(0, len(df_test) - 1, 100, dtype=int)
        test_sample = df_test.iloc[sample_indices]

        booster_ref = xgb.Booster()
        booster_ref.load_model(model_path)
        load_model_artifacts()

        feature_diffs = []
        pred_diffs = []

        for idx, (t, row) in enumerate(test_sample.iterrows()):
            vec_train = row[FEATURE_NAMES].values.astype(float).tolist()
            lag_24 = float(row["temperature_2m_lag_24"])

            # 1. Direct prediction from training feature row
            dmat = xgb.DMatrix(pd.DataFrame([row[FEATURE_NAMES]]), feature_names=FEATURE_NAMES)
            p_train = lag_24 + float(booster_ref.predict(dmat)[0])

            # 2. Production reconstruction
            raw_idx = df.index.get_loc(t)
            history_168 = df[TARGET].iloc[raw_idx-168:raw_idx].tolist()
            row_raw = df.loc[t]

            press_now = float(row_raw["pressure_msl"])
            p_ch3 = press_now - float(df["pressure_msl"].iloc[raw_idx - 3])
            p_ch6 = press_now - float(df["pressure_msl"].iloc[raw_idx - 6])

            exo = ExogenousWeather(
                apparent_temperature=float(row_raw["apparent_temperature"]),
                pressure_msl=press_now,
                relative_humidity_2m=float(row_raw["relative_humidity_2m"]),
                dew_point_2m=float(row_raw["dew_point_2m"]),
                wind_speed_10m=float(row_raw["wind_speed_10m"]),
                cloud_cover=float(row_raw["cloud_cover"]),
                pressure_change_3h=p_ch3,
                pressure_change_6h=p_ch6
            )

            vec_prod = build_production_features(history_168, t, exo)
            p_prod = predict_temperature(vec_prod, lag_24=lag_24)

            f_diff = max(abs(a - b) for a, b in zip(vec_train, vec_prod))
            feature_diffs.append(f_diff)
            pred_diffs.append(abs(p_train - p_prod))

        max_feat_diff = max(feature_diffs)
        max_pred_diff = max(pred_diffs)

        self.assertLessEqual(
            max_feat_diff,
            1e-6,
            f"FAIL: Maximum feature difference {max_feat_diff:.2e} exceeded tolerance 1e-6"
        )
        self.assertLessEqual(
            max_pred_diff,
            1e-6,
            f"FAIL: Maximum prediction difference {max_pred_diff:.2e} exceeded tolerance 1e-6"
        )
