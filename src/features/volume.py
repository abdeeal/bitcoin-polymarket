"""
Volume Features — 15 features, fully NumPy vectorized.

Target latency: <5ms for all 15 features.
NO Python for-loops. All operations are broadcasting/vectorized.
"""

import numpy as np
from src.features.vectorization_standards import (
    rolling_mean_np,
    rolling_std_np,
    compute_volume_ratio,
    compute_taker_buy_sell_ratio,
)
from src.features.jit_kernels import compute_volume_delta_cumulative


def compute_volume_features(w: dict) -> dict:
    """
    Compute all volume features from Ring Buffer window.

    Args:
        w: dict of numpy arrays from Ring Buffer get_window().

    Returns:
        dict of feature_name -> scalar value (latest bar).
    """
    vol = w['volume']
    tb = w['taker_buy']
    c = w['close']
    n = len(vol)

    if n < 5:
        return _empty_features()

    # --- Volume ratios (vs rolling mean) ---
    vratio_5 = float(compute_volume_ratio(vol, 5)[-1]) if n >= 5 else 1.0
    vratio_20 = float(compute_volume_ratio(vol, 20)[-1]) if n >= 20 else 1.0
    vratio_60 = float(compute_volume_ratio(vol, 60)[-1]) if n >= 60 else 1.0

    # --- Taker buy/sell ratio ---
    tb_ratio = float(compute_taker_buy_sell_ratio(tb, vol)[-1])

    # --- Rolling taker buy ratio (smoothed) ---
    taker_pct = np.where(vol == 0, 0.5, tb / vol)
    tb_pct_5 = float(rolling_mean_np(taker_pct, 5)[-1]) if n >= 5 else 0.5
    tb_pct_20 = float(rolling_mean_np(taker_pct, 20)[-1]) if n >= 20 else 0.5

    # --- Volume trend (slope of rolling mean) ---
    vol_ma_20 = rolling_mean_np(vol, 20) if n >= 20 else np.full(n, vol[-1])
    vol_trend = 0.0
    if n >= 25:
        vol_trend = float((vol_ma_20[-1] - vol_ma_20[-5]) / (vol_ma_20[-5] + 1e-10))

    # --- Volume volatility ---
    vol_std_20 = float(rolling_std_np(vol, 20)[-1]) if n >= 20 else 0.0
    vol_cv_20 = vol_std_20 / (float(rolling_mean_np(vol, 20)[-1]) + 1e-10) if n >= 20 else 0.0

    # --- On-Balance Volume change ---
    price_dir = np.sign(np.diff(c))
    obv_delta = np.sum(price_dir[-5:] * vol[-5:]) if n >= 6 else 0.0

    # --- Cumulative Volume Delta ---
    cvd_5 = float(compute_volume_delta_cumulative(tb, vol, 5)[-1]) if n >= 5 else 0.0
    cvd_20 = float(compute_volume_delta_cumulative(tb, vol, 20)[-1]) if n >= 20 else 0.0

    # --- Volume-price divergence ---
    # High volume + small move = absorption; Low volume + big move = breakout
    ret = np.abs(np.diff(c) / (c[:-1] + 1e-10))
    vol_norm = vol[1:] / (rolling_mean_np(vol, 20)[1:] + 1e-10) if n >= 20 else np.ones(n - 1)
    vp_divergence = float(vol_norm[-1] - ret[-1] * 100) if len(ret) > 0 else 0.0

    # --- Large trade ratio (from microstructure fields in Ring Buffer) ---
    large_trades = w.get('large_trade_count', np.zeros(n))
    trade_count = w.get('trade_count', np.ones(n))
    safe_tc = np.where(trade_count == 0, 1, trade_count)
    lt_ratio = float(large_trades[-1] / safe_tc[-1]) if n > 0 else 0.0

    return {
        'volume_ratio_5': vratio_5,
        'volume_ratio_20': vratio_20,
        'volume_ratio_60': vratio_60,
        'taker_bs_ratio': tb_ratio,
        'taker_buy_pct_5': tb_pct_5,
        'taker_buy_pct_20': tb_pct_20,
        'volume_trend_20': vol_trend,
        'volume_std_20': vol_std_20,
        'volume_cv_20': vol_cv_20,
        'obv_delta_5': float(obv_delta),
        'cvd_5': cvd_5,
        'cvd_20': cvd_20,
        'vp_divergence': vp_divergence,
        'large_trade_ratio': lt_ratio,
        'avg_trade_size': float(w.get('avg_trade_size', np.zeros(n))[-1]),
    }


def _empty_features() -> dict:
    keys = [
        'volume_ratio_5', 'volume_ratio_20', 'volume_ratio_60',
        'taker_bs_ratio', 'taker_buy_pct_5', 'taker_buy_pct_20',
        'volume_trend_20', 'volume_std_20', 'volume_cv_20',
        'obv_delta_5', 'cvd_5', 'cvd_20', 'vp_divergence',
        'large_trade_ratio', 'avg_trade_size',
    ]
    return {k: 0.0 for k in keys}
