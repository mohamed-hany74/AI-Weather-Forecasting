"""
Evaluation Metrics Module for Time-Series Regression.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard time-series forecast evaluation metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    
    # MAPE with epsilon protection
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "medae": round(medae, 6),
        "mape": round(mape, 4)
    }
