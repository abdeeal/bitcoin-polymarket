"""
Order Book Features — 20 features, NumPy + Numba JIT.

Target latency: <15ms for all 20 features.
Uses Numba JIT for OFI computation, NumPy for depth analysis.
"""

import numpy as np
from src.features.jit_kernels import compute_ofi, compute_ofi_rolling


def compute_order_book(w: dict) -> dict:
    """
    Compute all order book features from Ring Buffer window.

    Args:
        w: dict of numpy arrays from Ring Buffer get_window().

    Returns:
        dict of feature_name -> scalar value (latest bar).
    """
    n = len(w['close'])

    if n < 5:
        return _empty_features()

    bid_d = w.get('bid_depth', np.zeros(n))
    ask_d = w.get('ask_depth', np.zeros(n))
    spread = w.get('spread', np.zeros(n))
    mid = w.get('mid_price', w['close'])
    bid_1pct = w.get('bid_depth_1pct', np.zeros(n))
    ask_1pct = w.get('ask_depth_1pct', np.zeros(n))
    ba_ratio_5 = w.get('bid_ask_ratio_5', np.zeros(n))
    ba_ratio_10 = w.get('bid_ask_ratio_10', np.zeros(n))

    # --- Bid-Ask Depth Imbalance ---
    total_depth = bid_d + ask_d
    safe_total = np.where(total_depth == 0, 1e-10, total_depth)
    depth_imbalance = (bid_d - ask_d) / safe_total

    # --- OFI (Order Flow Imbalance) via Numba JIT ---
    # Use mid_price as proxy for bid/ask prices if direct OB prices unavailable
    ofi_val = compute_ofi(bid_d, ask_d, mid, mid + spread)

    # Rolling OFI
    ofi_5 = 0.0
    ofi_20 = 0.0
    if n >= 5:
        ofi_roll = compute_ofi_rolling(bid_d, ask_d, mid, mid + spread, 5)
        ofi_5 = float(ofi_roll[-1]) if not np.isnan(ofi_roll[-1]) else 0.0
    if n >= 20:
        ofi_roll_20 = compute_ofi_rolling(bid_d, ask_d, mid, mid + spread, 20)
        ofi_20 = float(ofi_roll_20[-1]) if not np.isnan(ofi_roll_20[-1]) else 0.0

    # --- Spread analysis ---
    spread_bps = spread / (mid + 1e-10) * 10000  # in basis points
    spread_mean_20 = float(np.mean(spread_bps[-20:])) if n >= 20 else float(spread_bps[-1])
    spread_z = 0.0
    if n >= 20:
        sp_std = float(np.std(spread_bps[-20:]))
        if sp_std > 1e-10:
            spread_z = float((spread_bps[-1] - spread_mean_20) / sp_std)

    # --- Depth at 1% levels ---
    depth_1pct_ratio = np.where(
        ask_1pct == 0, 0.0, bid_1pct / (ask_1pct + 1e-10)
    )

    # --- Depth change rate ---
    bid_depth_change = 0.0
    ask_depth_change = 0.0
    if n >= 2:
        bid_depth_change = float((bid_d[-1] - bid_d[-2]) / (bid_d[-2] + 1e-10))
        ask_depth_change = float((ask_d[-1] - ask_d[-2]) / (ask_d[-2] + 1e-10))

    # --- Depth disparity (asymmetry signal) ---
    depth_disparity = 0.0
    if n >= 5:
        bid_mean_5 = float(np.mean(bid_d[-5:]))
        ask_mean_5 = float(np.mean(ask_d[-5:]))
        depth_disparity = (bid_mean_5 - ask_mean_5) / (bid_mean_5 + ask_mean_5 + 1e-10)

    return {
        'depth_imbalance': float(depth_imbalance[-1]),
        'depth_imbalance_5m_avg': float(np.mean(depth_imbalance[-5:])) if n >= 5 else 0.0,
        'ofi': float(ofi_val),
        'ofi_rolling_5': ofi_5,
        'ofi_rolling_20': ofi_20,
        'spread_bps': float(spread_bps[-1]),
        'spread_mean_20': spread_mean_20,
        'spread_zscore': spread_z,
        'bid_ask_ratio_5': float(ba_ratio_5[-1]),
        'bid_ask_ratio_10': float(ba_ratio_10[-1]),
        'depth_1pct_ratio': float(depth_1pct_ratio[-1]),
        'bid_depth_change': bid_depth_change,
        'ask_depth_change': ask_depth_change,
        'depth_disparity_5': depth_disparity,
        'total_depth': float(total_depth[-1]),
        'total_depth_change': float(
            (total_depth[-1] - total_depth[-2]) / (total_depth[-2] + 1e-10)
        ) if n >= 2 else 0.0,
        'bid_depth': float(bid_d[-1]),
        'ask_depth': float(ask_d[-1]),
        'mid_price': float(mid[-1]),
        'spread_raw': float(spread[-1]),
    }


def _empty_features() -> dict:
    keys = [
        'depth_imbalance', 'depth_imbalance_5m_avg',
        'ofi', 'ofi_rolling_5', 'ofi_rolling_20',
        'spread_bps', 'spread_mean_20', 'spread_zscore',
        'bid_ask_ratio_5', 'bid_ask_ratio_10',
        'depth_1pct_ratio', 'bid_depth_change', 'ask_depth_change',
        'depth_disparity_5', 'total_depth', 'total_depth_change',
        'bid_depth', 'ask_depth', 'mid_price', 'spread_raw',
    ]
    return {k: 0.0 for k in keys}
