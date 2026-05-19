"""
Vectorization Standards — Mandatory rules for all feature computation.

EVERY feature function in this codebase MUST follow these rules:
  1. NO Python for-loops on arrays in any hot-path computation
  2. Use pure NumPy vectorized operations or Numba @njit kernels
  3. Avoid pandas — NumPy arrays from Ring Buffer are the only input
  4. Handle div-by-zero via np.where (vectorized), never try/except

This file provides validated utility functions used across all feature modules.
"""

import numpy as np


# =============================================================================
# Return Computations
# =============================================================================

def compute_returns(close: np.ndarray) -> np.ndarray:
    """
    Simple returns: (close[i] - close[i-1]) / close[i-1]

    ✅ Fully vectorized — single np.diff + division.
    """
    prev = close[:-1]
    prev = np.where(prev == 0, 1e-10, prev)
    return np.diff(close) / prev


def compute_log_returns(close: np.ndarray) -> np.ndarray:
    """
    Log returns: log(close[i] / close[i-1])

    ✅ Fully vectorized — single np.log + np.diff.
    """
    safe_close = np.where(close <= 0, 1e-10, close)
    return np.diff(np.log(safe_close))


# =============================================================================
# Candle Body Analysis
# =============================================================================

def compute_body_ratio(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Body-to-range ratio: (close - open) / (high - low).

    Measures how much of the candle range is "body" vs wicks.
    Values near +1/-1 = strong directional, near 0 = doji/spinning top.

    ✅ Fully vectorized.
    """
    hl = high - low
    hl = np.where(hl == 0, 1e-10, hl)
    return (close - open_) / hl


def compute_upper_wick_ratio(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Upper wick ratio: (high - max(open, close)) / (high - low).

    ✅ Fully vectorized.
    """
    hl = high - low
    hl = np.where(hl == 0, 1e-10, hl)
    body_top = np.maximum(open_, close)
    return (high - body_top) / hl


def compute_lower_wick_ratio(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Lower wick ratio: (min(open, close) - low) / (high - low).

    ✅ Fully vectorized.
    """
    hl = high - low
    hl = np.where(hl == 0, 1e-10, hl)
    body_bottom = np.minimum(open_, close)
    return (body_bottom - low) / hl


# =============================================================================
# Rolling Window Operations (Pure NumPy — No Loops)
# =============================================================================

def rolling_sum_np(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling sum using cumulative sum trick.

    ✅ O(n) — single cumsum + subtraction, no loop.
    """
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    cumsum = np.cumsum(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan
    result[window - 1] = cumsum[window - 1]
    result[window:] = cumsum[window:] - cumsum[:-window]
    return result


def rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean using cumulative sum trick.

    ✅ O(n) — single cumsum + division, no loop.
    """
    return rolling_sum_np(arr, window) / window


def rolling_max_np(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling maximum using stride tricks.

    ✅ Vectorized via as_strided + np.max.
    """
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    maxs = np.max(windows, axis=1)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan
    result[window - 1:] = maxs
    return result


def rolling_min_np(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling minimum using stride tricks.

    ✅ Vectorized via as_strided + np.min.
    """
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    mins = np.min(windows, axis=1)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan
    result[window - 1:] = mins
    return result


def rolling_std_np(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation using stride tricks.

    ✅ Vectorized via as_strided + np.std.
    """
    n = len(arr)
    if n < window:
        return np.full(n, np.nan)
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    stds = np.std(windows, axis=1)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan
    result[window - 1:] = stds
    return result


# =============================================================================
# Volume Analysis (Vectorized)
# =============================================================================

def compute_volume_ratio(volume: np.ndarray, window: int) -> np.ndarray:
    """
    Current volume / rolling mean volume.

    ✅ Fully vectorized.
    """
    mean_vol = rolling_mean_np(volume, window)
    mean_vol = np.where(mean_vol == 0, 1e-10, mean_vol)
    result = volume / mean_vol
    return result


def compute_taker_buy_sell_ratio(
    taker_buy: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Taker buy/sell ratio: taker_buy / (volume - taker_buy + epsilon).

    ✅ Fully vectorized.
    """
    taker_sell = volume - taker_buy
    taker_sell = np.where(taker_sell == 0, 1e-10, taker_sell)
    return taker_buy / taker_sell


# =============================================================================
# Price Position Features (Vectorized)
# =============================================================================

def price_position_in_range(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Position of current price within rolling high-low range.

    Value: 0 = at rolling low, 1 = at rolling high.

    ✅ Fully vectorized using rolling_max/rolling_min.
    """
    roll_high = rolling_max_np(high, window)
    roll_low = rolling_min_np(low, window)
    rng = roll_high - roll_low
    rng = np.where(rng == 0, 1e-10, rng)
    return (close - roll_low) / rng


def distance_from_ema(
    close: np.ndarray,
    ema: np.ndarray,
) -> np.ndarray:
    """
    Percentage distance of close from EMA.

    ✅ Fully vectorized.
    """
    safe_ema = np.where(ema == 0, 1e-10, ema)
    return (close - ema) / safe_ema


# =============================================================================
# EMA (Exponential Moving Average — vectorized via cumulative approach)
# =============================================================================

def compute_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential Moving Average — iterative but necessary for EMA semantics.

    Note: EMA is inherently sequential (each value depends on the previous).
    This is one of the few places where a loop is unavoidable in Python.
    For production, TA-Lib's C implementation is used instead.
    This function exists as a fallback and for testing.

    Args:
        arr:  Input price array.
        span: EMA span (e.g. 5, 10, 20).

    Returns:
        Array of EMA values.
    """
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result
