"""
Feature Normalizer — Rolling z-score and min-max normalization.
All operations vectorized via NumPy. No Python for-loops.
"""
import numpy as np

def zscore_normalize(value: float, arr: np.ndarray, window: int) -> float:
    """Single-bar z-score using last N values. No loop."""
    w = arr[-window:]
    mean = float(w.mean()); std = float(w.std())
    return (value - mean) / (std + 1e-10)

def minmax_normalize(value: float, arr: np.ndarray, window: int) -> float:
    """Single-bar min-max scale using last N values. No loop."""
    w = arr[-window:]
    mn = float(w.min()); mx = float(w.max())
    return (value - mn) / (mx - mn + 1e-10)

def batch_zscore(features: dict, history: dict, window: int = 100) -> dict:
    """Z-score normalize all features using their rolling history."""
    result = {}
    for key, value in features.items():
        if key in ('bot_state',) or not isinstance(value, (int, float)):
            result[key] = value
            continue
        if key in history and len(history[key]) >= window:
            arr = np.array(history[key][-window:], dtype=np.float64)
            result[f'{key}_z'] = zscore_normalize(value, arr, window)
        result[key] = value
    return result
