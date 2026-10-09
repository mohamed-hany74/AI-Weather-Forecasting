"""
Model Service: Thread-safe singleton for XGBoost model inference & metadata inspection.
"""

import json
import logging
import math
import threading
from pathlib import Path
from typing import List, Dict, Any
from django.conf import settings
from xgboost import XGBRegressor

from ..domain.exceptions import ModelInferenceError
from ..domain.contracts import FEATURE_NAMES, REQUIRED_FEATURE_COUNT

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

                # Load XGBoost booster
                model = XGBRegressor()
                model.load_model(str(model_file))
                _model_instance = model

                # Load feature config strictly
                if not config_file.exists():
                    raise ModelInferenceError(f"Feature config file not found: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    _feature_config = json.load(f)

                # Load metadata strictly - fail fast if missing
                if not meta_file.exists():
                    raise ModelInferenceError(f"Model metadata file not found: {meta_file}")
                with open(meta_file, 'r', encoding='utf-8') as f:
                    _metadata = json.load(f)

                logger.info(f"Loaded XGBoost model from {model_file} with {len(FEATURE_NAMES)} features.")


def is_model_ready() -> bool:
    """Check if model is initialized and expects exactly 15 features."""
    try:
        load_model_artifacts()
        if _model_instance is None or _metadata is None:
            return False
        return len(FEATURE_NAMES) == REQUIRED_FEATURE_COUNT
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False


def predict_temperature(feature_vector: List[float]) -> float:
    """Run model inference on pre-ordered 15-element feature vector and validate output."""
    load_model_artifacts()
    if len(feature_vector) != REQUIRED_FEATURE_COUNT:
        raise ModelInferenceError(
            f"Expected {REQUIRED_FEATURE_COUNT} features, got {len(feature_vector)}"
        )
    try:
        pred_raw = _model_instance.predict([feature_vector])[0]
        pred = float(pred_raw)
    except Exception as e:
        logger.error(f"XGBoost predict failure: {e}", exc_info=True)
        raise ModelInferenceError(f"Model prediction failed: {str(e)}")

    # Strict Output Validation: Guarantee finite number (no NaN, Inf, -Inf)
    if not math.isfinite(pred):
        logger.error(f"CRITICAL: Model generated non-finite output: {pred}. Input vector: {feature_vector}")
        raise ModelInferenceError(f"Non-finite model prediction generated ({pred}).")

    return pred


def get_safe_model_info() -> Dict[str, Any]:
    """Retrieve verified metadata for API presentation without fallback masking."""
    load_model_artifacts()
    if "test_mae" not in _metadata:
        raise ModelInferenceError("Metadata does not contain required metric 'test_mae'.")

    return {
        "model_type": _metadata["model_type"],
        "forecast_horizon_hours": _metadata["forecast_horizon_hours"],
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "metrics": {
            "test_mae": _metadata["test_mae"],
            "test_rmse": _metadata["test_rmse"],
            "test_r2": _metadata["test_r2"],
            "test_mape": _metadata["test_mape"]
        },
        "dependencies": _metadata.get("dependencies", {})
    }

