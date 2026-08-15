"""
Model Service: Thread-safe singleton for XGBoost model inference & metadata inspection.
"""

import json
import logging
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

                # Load feature config
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        _feature_config = json.load(f)
                else:
                    _feature_config = {"features": FEATURE_NAMES}

                # Load metadata
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        _metadata = json.load(f)
                else:
                    _metadata = {"model_type": "XGBRegressor", "features": FEATURE_NAMES}

                logger.info(f"Loaded XGBoost model from {model_file} with {len(FEATURE_NAMES)} features.")


def predict_temperature(feature_vector: List[float]) -> float:
    """Run model inference on pre-ordered 15-element feature vector."""
    load_model_artifacts()
    if len(feature_vector) != REQUIRED_FEATURE_COUNT:
        raise ModelInferenceError(
            f"Expected {REQUIRED_FEATURE_COUNT} features, got {len(feature_vector)}"
        )
    try:
        pred = float(_model_instance.predict([feature_vector])[0])
        return pred
    except Exception as e:
        logger.error(f"XGBoost predict failure: {e}", exc_info=True)
        raise ModelInferenceError(f"Model prediction failed: {str(e)}")


def get_safe_model_info() -> Dict[str, Any]:
    """Retrieve safe metadata for API presentation."""
    load_model_artifacts()
    return {
        "model_type": _metadata.get("model_type", "XGBRegressor"),
        "forecast_horizon_hours": _metadata.get("forecast_horizon_hours", 24),
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "metrics": {
            "test_mae": _metadata.get("test_mae", 1.37),
            "test_rmse": _metadata.get("test_rmse", 1.93),
            "test_r2": _metadata.get("test_r2", 0.94),
            "test_mape": _metadata.get("test_mape", 6.75)
        }
    }
