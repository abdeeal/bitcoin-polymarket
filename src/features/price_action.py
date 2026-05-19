"""
Price Action Features — 20 features, fully NumPy vectorized.

Target latency: <5ms for all 20 features.
NO Python for-loops. All operations are broadcasting/vectorized.
"""

import numpy as np
from src.features.vectorization_standards import (
    compute_returns,
    compute_log_returns,
    compute_body_ratio,
    compute_upper_wick_ratio,
    compute_lower_wick_ratio,
    rolling_mean_np,
    rolling_max_np,
    rolling_min_np,
    rolling_std_np,
    price_position_in_range,
)
from src.features.jit_kernels import compute_consecutive_count


def compute_price_action(w: dict) -> dict:
    """
    Compute all price action features from Ring Buffer window.

    Args:
        w: dict of numpy arrays from Ring Buffer get_window().

    Returns:
        dict of feature_name -> scalar value (latest bar).
    """
    c = w['close']
    o = w['open']
    h = w['high']
    l = w['low']
    n = len(c)

    if n < 5:
        return _empty_features()

    # --- Returns ---
    returns = compute_returns(c)
    log_ret = compute_log_returns(c)

    # --- Candle structure ---
    body = compute_body_ratio(o, h, l, c)
    upper_wick = compute_upper_wick_ratio(o, h, l, c)
    lower_wick = compute_lower_wick_ratio(o, h, l, c)

    # --- Multi-timeframe returns ---
    ret_1 = returns[-1] if len(returns) > 0 else 0.0
    ret_5 = (c[-1] / c[-6] - 1) if n >= 6 else 0.0
    ret_10 = (c[-1] / c[-11] - 1) if n >= 11 else 0.0
    ret_20 = (c[-1] / c[-21] - 1) if n >= 21 else 0.0
    ret_60 = (c[-1] / c[-61] - 1) if n >= 61 else 0.0

    # --- Realized volatility ---
    vol_5 = float(rolling_std_np(returns, 5)[-1]) if len(returns) >= 5 else 0.0
    vol_20 = float(rolling_std_np(returns, 20)[-1]) if len(returns) >= 20 else 0.0
    vol_60 = float(rolling_std_np(returns, 60)[-1]) if len(returns) >= 60 else 0.0

    # --- Price position in range ---
    pos_20 = float(price_position_in_range(c, h, l, 20)[-1]) if n >= 20 else 0.5
    pos_60 = float(price_position_in_range(c, h, l, 60)[-1]) if n >= 60 else 0.5

    # --- High/Low distance ---
    roll_high_20 = float(rolling_max_np(h, 20)[-1]) if n >= 20 else h[-1]
    roll_low_20 = float(rolling_min_np(l, 20)[-1]) if n >= 20 else l[-1]
    dist_from_high = (c[-1] - roll_high_20) / (roll_high_20 + 1e-10)
    dist_from_low = (c[-1] - roll_low_20) / (roll_low_20 + 1e-10)

    # --- Consecutive green/red candles ---
    directions = np.sign(c - o).astype(np.float64)
    consec_green = float(compute_consecutive_count(directions, 1.0)[-1])
    consec_red = float(compute_consecutive_count(directions, -1.0)[-1])

    return {
        'return_1m': float(ret_1),
        'return_5m': float(ret_5),
        'return_10m': float(ret_10),
        'return_20m': float(ret_20),
        'return_60m': float(ret_60),
        'log_return_1m': float(log_ret[-1]) if len(log_ret) > 0 else 0.0,
        'body_ratio': float(body[-1]),
        'upper_wick_ratio': float(upper_wick[-1]),
        'lower_wick_ratio': float(lower_wick[-1]),
        'realized_vol_5': vol_5,
        'realized_vol_20': vol_20,
        'realized_vol_60': vol_60,
        'price_position_20': pos_20,
        'price_position_60': pos_60,
        'dist_from_high_20': float(dist_from_high),
        'dist_from_low_20': float(dist_from_low),
        'consecutive_green': consec_green,
        'consecutive_red': consec_red,
        'range_pct': float((h[-1] - l[-1]) / (c[-1] + 1e-10)),
        'gap': float(o[-1] - c[-2]) if n >= 2 else 0.0,
    }


def _empty_features() -> dict:
    """Return zeroed features when insufficient data."""
    keys = [
        'return_1m', 'return_5m', 'return_10m', 'return_20m', 'return_60m',
        'log_return_1m', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'realized_vol_5', 'realized_vol_20', 'realized_vol_60',
        'price_position_20', 'price_position_60',
        'dist_from_high_20', 'dist_from_low_20',
        'consecutive_green', 'consecutive_red', 'range_pct', 'gap',
    ]
    return {k: 0.0 for k in keys}
