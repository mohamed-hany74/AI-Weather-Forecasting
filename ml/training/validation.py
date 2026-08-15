"""
Validation Rules for ML Datasets and Temporal Slices.
"""

import numpy as np
import pandas as pd
from typing import List


def validate_feature_matrix(X: pd.DataFrame, expected_features: List[str]) -> None:
    """Validate feature matrix schema, order, and finite properties."""
    if list(X.columns) != expected_features:
        raise ValueError(f"Feature order mismatch! Expected: {expected_features}, got: {list(X.columns)}")
    
    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values found in feature columns: {nan_cols}")
        
    if np.isinf(X.values).any():
        inf_cols = X.columns[np.isinf(X.values).any(axis=0)].tolist()
        raise ValueError(f"Infinite values found in feature columns: {inf_cols}")


def validate_target_series(y: pd.Series) -> None:
    """Validate target series values."""
    if y.isna().any():
        raise ValueError("NaN values found in target series.")
    if np.isinf(y.values).any():
        raise ValueError("Infinite values found in target series.")


def validate_temporal_boundary(train_last_t: pd.Timestamp, test_first_t: pd.Timestamp, horizon_hours: int = 24) -> None:
    """Verify strict purge/embargo cutoff between train set target and test set features."""
    last_train_target_t = train_last_t + pd.Timedelta(hours=horizon_hours)
    if last_train_target_t >= test_first_t:
        raise ValueError(
            f"Temporal boundary leakage detected! Last train target at {last_train_target_t} >= First test feature at {test_first_t}"
        )
