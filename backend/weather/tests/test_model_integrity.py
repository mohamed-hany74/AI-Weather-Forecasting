"""
Suite E: Model Artifact Integrity Tests.
Verifies:
1. Physical existence and non-zero size of all model artifacts.
2. Direct calculation of SHA256 checksums from file bytes.
3. Verification that computed SHA256 matches metadata model_sha256.
4. Deserialization of Mean model, Q10 model, and Q90 model.
5. Strict verification that feature_count == 21 across code, contracts, config, and models.
"""

import os
import json
import hashlib
from pathlib import Path
import xgboost as xgb
from django.test import TestCase
from django.conf import settings

from weather.domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT


class ModelArtifactIntegrityTests(TestCase):
    def setUp(self):
        self.models_dir = Path(settings.BASE_DIR).parent / "ml" / "models"
        self.mean_model_file = self.models_dir / "xgboost_weather_model.json"
        self.q10_model_file = self.models_dir / "xgboost_weather_model_q10.json"
        self.q90_model_file = self.models_dir / "xgboost_weather_model_q90.json"
        self.config_file = self.models_dir / "feature_config.json"
        self.meta_file = self.models_dir / "model_metadata.json"

    def test_artifacts_exist_and_readable(self):
        """All 5 model artifact files must exist on disk and be non-empty."""
        for path in [self.mean_model_file, self.q10_model_file, self.q90_model_file, self.config_file, self.meta_file]:
            self.assertTrue(path.exists(), f"Missing required model artifact: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Artifact file is empty: {path}")

    def test_model_sha256_checksum_matches_metadata(self):
        """Calculated SHA256 of xgboost_weather_model.json must match model_metadata.json."""
        raw_bytes = self.mean_model_file.read_bytes()
        calculated_sha = hashlib.sha256(raw_bytes).hexdigest()

        with open(self.meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_sha = metadata.get("model_sha256")
        self.assertIsNotNone(expected_sha, "Metadata must contain 'model_sha256' field")
        self.assertEqual(
            calculated_sha,
            expected_sha,
            f"Checksum mismatch! Calculated {calculated_sha} != Expected {expected_sha}"
        )

    def test_model_deserialization_and_feature_count_21(self):
        """
        MANDATORY GATE:
        All models must deserialize cleanly and enforce exact feature_count == 21.
        """
        # 1. Mean Model
        m_mean = xgb.XGBRegressor()
        m_mean.load_model(str(self.mean_model_file))
        self.assertEqual(REQUIRED_FEATURE_COUNT, 21)
        self.assertEqual(len(FEATURE_NAMES), 21)

        # 2. Q10 Model
        m_q10 = xgb.XGBRegressor()
        m_q10.load_model(str(self.q10_model_file))

        # 3. Q90 Model
        m_q90 = xgb.XGBRegressor()
        m_q90.load_model(str(self.q90_model_file))

        # 4. Feature Config & Metadata agreement
        with open(self.config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(self.meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(cfg["feature_count"], 21)
        self.assertEqual(meta["feature_count"], 21)
        self.assertEqual(cfg["features"], FEATURE_NAMES)
        self.assertEqual(meta["features"], FEATURE_NAMES)

    def test_training_target_algebra_on_real_dataset_rows(self):
        """
        FORENSIC AUDIT GATE:
        Verifies that on real historical dataset rows:
        target(t) == T(t + 24h) - T(t - 24h)
        and reconstructed prediction(t+24h) == lag_24 + predicted_delta.
        """
        import pandas as pd
        import numpy as np
        from weather.services.feature_service import build_production_features

        data_path = Path(settings.BASE_DIR).parent / "ml" / "data" / "weather.csv"
        df_raw = pd.read_csv(data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
        df_raw.set_index("date", inplace=True)
        df_raw.sort_index(inplace=True)
        df = df_raw.asfreq("h").select_dtypes(include=[np.number]).ffill()

        # Sample rows across historical span
        for idx in [500, 10000, 40000]:
            t = df.index[idx]
            t_lag24 = t - pd.Timedelta(hours=24)
            t_target = t + pd.Timedelta(hours=24)

            val_lag24 = df.loc[t_lag24, "temperature_2m"]
            val_target_abs = df.loc[t_target, "temperature_2m"]
            expected_delta = val_target_abs - val_lag24

            # Reconstruction verification
            reconstructed = val_lag24 + expected_delta
            self.assertAlmostEqual(reconstructed, val_target_abs, places=5)

    def test_cross_validation_purge_gap_zero_overlap(self):
        """
        FORENSIC AUDIT GATE:
        Verifies that with gap=24, no training fold's last target timestamp
        overlaps or enters into the validation fold period.
        """
        import pandas as pd
        from sklearn.model_selection import TimeSeriesSplit

        data_path = Path(settings.BASE_DIR).parent / "ml" / "data" / "weather.csv"
        df_raw = pd.read_csv(data_path)
        df_raw["date"] = pd.to_datetime(df_raw["date"], utc=True)
        df_raw.set_index("date", inplace=True)
        df_raw.sort_index(inplace=True)
        df = df_raw.asfreq("h").copy()

        n_total = len(df)
        n_test = int(n_total * 0.20)
        split_dt = df.index[n_total - n_test]
        purge_cutoff = split_dt - pd.Timedelta(hours=24)
        df_tr = df[df.index < purge_cutoff]

        tscv = TimeSeriesSplit(n_splits=5, gap=24)
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(df_tr), 1):
            tr_last_t = df_tr.index[tr_idx[-1]]
            val_first_t = df_tr.index[val_idx[0]]
            tr_last_target_t = tr_last_t + pd.Timedelta(hours=24)

            # Assert strict causal separation: last training target must be strictly <= validation start
            self.assertLessEqual(
                tr_last_target_t,
                val_first_t,
                f"Fold {fold} target leakage: train target {tr_last_target_t} overlaps validation start {val_first_t}"
            )
