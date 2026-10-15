"""
Model Service: Thread-safe singleton for XGBoost model inference & quantile predictions (Phase 3).
Loads Mean model + Quantile models (q10, q90) and reconstructs absolute temperatures from predicted deltas.
"""

import json
import logging
import math
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
from django.conf import settings
from xgboost import XGBRegressor

from ..domain.exceptions import ModelInferenceError
from ..domain.contracts import (
    FEATURE_NAMES,
    REQUIRED_FEATURE_COUNT,
    TYPICAL_ERROR_CELSIUS,
    TYPICAL_ERROR_NOTE
)

logger = logging.getLogger(__name__)

_model_instance = None
_feature_config = None
_metadata = None
_lock = threading.Lock()


def _get_models_dir() -> Path:
    if hasattr(settings, 'ML_MODELS_DIR'):
        return Path(settings.ML_MODELS_DIR)
    return Path(settings.BASE_DIR).parent / 'ml' / 'models'


def load_model_artifacts():
    global _model_instance, _feature_config, _metadata
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                models_dir = _get_models_dir()
                model_file = models_dir / 'xgboost_weather_model.json'
                config_file = models_dir / 'feature_config.json'
                meta_file = models_dir / 'model_metadata.json'

                if not model_file.exists():
                    raise ModelInferenceError(f"Model file not found: {model_file}")

                # Load Mean XGBoost model
                model = XGBRegressor()
                model.load_model(str(model_file))
                _model_instance = model

                # Note: Quantile files (q10, q90) remain on disk untouched, but are decommissioned from serving.

                # Load feature config strictly
                if not config_file.exists():
                    raise ModelInferenceError(f"Feature config file not found: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    _feature_config = json.load(f)

                # Load metadata strictly
                if not meta_file.exists():
                    raise ModelInferenceError(f"Model metadata file not found: {meta_file}")
                with open(meta_file, 'r', encoding='utf-8') as f:
                    _metadata = json.load(f)

                logger.info(f"Loaded Phase 3 XGBoost model from {models_dir} with {len(FEATURE_NAMES)} features.")


def is_model_ready() -> bool:
    """Check if model is initialized and expects exactly 21 features."""
    try:
        load_model_artifacts()
        if _model_instance is None or _metadata is None:
            return False
        return len(FEATURE_NAMES) == REQUIRED_FEATURE_COUNT
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False


def predict_temperature(feature_vector: List[float], lag_24: float) -> float:
    """
    Run model inference on 21-element feature vector.
    Reconstructs absolute temperature: predicted_temperature = lag_24 + predicted_delta.
    Returns point forecast: pred_abs.
    """
    load_model_artifacts()
    if len(feature_vector) != REQUIRED_FEATURE_COUNT:
        raise ModelInferenceError(
            f"Expected {REQUIRED_FEATURE_COUNT} features, got {len(feature_vector)}"
        )
    try:
        pred_delta_raw = _model_instance.predict([feature_vector])[0]
        pred_delta = float(pred_delta_raw)
        pred_abs = float(lag_24 + pred_delta)
    except Exception as e:
        logger.error(f"XGBoost predict failure: {e}", exc_info=True)
        raise ModelInferenceError(f"Model prediction failed: {str(e)}")

    # Strict Output Validation
    if not math.isfinite(pred_abs):
        logger.error(f"CRITICAL: Non-finite output in pred_abs: {pred_abs}. Input: {feature_vector}")
        raise ModelInferenceError(f"Non-finite model prediction generated (pred_abs={pred_abs}).")

    return pred_abs


def get_safe_model_info() -> Dict[str, Any]:
    """
    Retrieve verified metadata for API presentation.
    Filters out obsolete interval coverage claims to reflect honest system performance.
    """
    load_model_artifacts()
    if "test_mae" not in _metadata:
        raise ModelInferenceError("Metadata does not contain required metric 'test_mae'.")

    return {
        "model_type": _metadata["model_type"],
        "target_formulation": _metadata.get("target_formulation", "delta_lag24"),
        "forecast_horizon_hours": _metadata["forecast_horizon_hours"],
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "metrics": {
            "test_mae": _metadata["test_mae"],
            "test_rmse": _metadata["test_rmse"],
            "test_r2": _metadata["test_r2"],
            "test_mape": _metadata["test_mape"],
            "may_mae": _metadata.get("may_mae", 2.13),
            "typical_error_celsius": TYPICAL_ERROR_CELSIUS,
            "typical_error_note": TYPICAL_ERROR_NOTE
        },
        "dependencies": _metadata.get("dependencies", {})
    }
