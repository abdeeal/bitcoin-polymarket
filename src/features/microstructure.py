"""
Microstructure Features — 15 features, Numba JIT kernels.

Target latency: <20ms for all 15 features.
Uses Numba JIT for Kyle's Lambda, Amihud illiquidity, and VWAP.
"""

import numpy as np
from src.features.vectorization_standards import compute_returns
from src.features.jit_kernels import (
    compute_kyle_lambda,
    compute_amihud,
    compute_vwap_rolling,
    percentile_rank_rolling,
    compute_rolling_correlation,
)


def compute_microstructure(w: dict) -> dict:
    """
    Compute all microstructure features from Ring Buffer window.

    These features capture market microstructure dynamics:
      - Kyle's Lambda (market impact / liquidity)
      - Amihud ratio (illiquidity)
      - VWAP deviation
      - Volume-price correlation
      - Trade size distribution metrics

    Args:
        w: dict of numpy arrays from Ring Buffer get_window().

    Returns:
        dict of feature_name -> scalar value (latest bar).
    """
    c = w['close']
    vol = w['volume']
    tb = w['taker_buy']
    n = len(c)

    if n < 10:
        return _empty_features()

    # --- Returns for microstructure ---
    returns = compute_returns(c)

    # Need padded arrays (returns is n-1 length, align to volume)
    ret_padded = np.empty(n, dtype=np.float64)
    ret_padded[0] = 0.0
    ret_padded[1:] = returns

    # --- Kyle's Lambda (market impact) ---
    kyle_10 = float(compute_kyle_lambda(ret_padded, vol, 10)[-1]) if n >= 10 else 0.0
    kyle_20 = float(compute_kyle_lambda(ret_padded, vol, 20)[-1]) if n >= 20 else 0.0
    kyle_60 = float(compute_kyle_lambda(ret_padded, vol, 60)[-1]) if n >= 60 else 0.0

    # Kyle's Lambda percentile rank (contextualizes within history)
    kyle_prank = 0.5
    if n >= 60:
        kyle_arr = compute_kyle_lambda(ret_padded, vol, 20)
        valid_kyle = kyle_arr[~np.isnan(kyle_arr)]
        if len(valid_kyle) >= 20:
            prank = percentile_rank_rolling(valid_kyle, 20)
            kyle_prank = float(prank[-1]) if not np.isnan(prank[-1]) else 0.5

    # --- Amihud Illiquidity Ratio ---
    amihud_10 = float(compute_amihud(ret_padded, vol, 10)[-1]) if n >= 10 else 0.0
    amihud_20 = float(compute_amihud(ret_padded, vol, 20)[-1]) if n >= 20 else 0.0
    amihud_60 = float(compute_amihud(ret_padded, vol, 60)[-1]) if n >= 60 else 0.0

    # --- VWAP and Deviation ---
    vwap_20 = float(compute_vwap_rolling(c, vol, 20)[-1]) if n >= 20 else float(c[-1])
    vwap_60 = float(compute_vwap_rolling(c, vol, 60)[-1]) if n >= 60 else float(c[-1])

    vwap_dev_20 = float((c[-1] - vwap_20) / (vwap_20 + 1e-10))
    vwap_dev_60 = float((c[-1] - vwap_60) / (vwap_60 + 1e-10))

    # --- Volume-price correlation (rolling) ---
    vp_corr = 0.0
    if n >= 20:
        corr = compute_rolling_correlation(ret_padded, vol, 20)
        vp_corr = float(corr[-1]) if not np.isnan(corr[-1]) else 0.0

    # --- Taker buy ratio trends ---
    taker_ratio = np.where(vol == 0, 0.5, tb / vol)
    taker_mean_5 = float(np.mean(taker_ratio[-5:])) if n >= 5 else 0.5
    taker_mean_20 = float(np.mean(taker_ratio[-20:])) if n >= 20 else 0.5
    taker_momentum = taker_mean_5 - taker_mean_20

    return {
        'kyle_lambda_10': kyle_10,
        'kyle_lambda_20': kyle_20,
        'kyle_lambda_60': kyle_60,
        'kyle_prank': kyle_prank,
        'amihud_10': amihud_10,
        'amihud_20': amihud_20,
        'amihud_60': amihud_60,
        'vwap_dev_20': vwap_dev_20,
        'vwap_dev_60': vwap_dev_60,
        'vp_correlation_20': vp_corr,
        'taker_ratio': float(taker_ratio[-1]),
        'taker_mean_5': taker_mean_5,
        'taker_mean_20': taker_mean_20,
        'taker_momentum': taker_momentum,
        'trade_count': float(w.get('trade_count', np.zeros(n))[-1]),
    }


def _empty_features() -> dict:
    keys = [
        'kyle_lambda_10', 'kyle_lambda_20', 'kyle_lambda_60', 'kyle_prank',
        'amihud_10', 'amihud_20', 'amihud_60',
        'vwap_dev_20', 'vwap_dev_60',
        'vp_correlation_20',
        'taker_ratio', 'taker_mean_5', 'taker_mean_20', 'taker_momentum',
        'trade_count',
    ]
    return {k: 0.0 for k in keys}
