"""
Lagged & Interaction Features — 15 features, fully NumPy vectorized.
Target latency: <10ms.
"""
import numpy as np

def compute_lagged_interaction(w: dict, base_features: dict) -> dict:
    n = len(w['close'])
    if n < 10: return _empty_features()

    ret1 = base_features.get('return_1m', 0.0)
    ret5 = base_features.get('return_5m', 0.0)
    vol_ratio = base_features.get('volume_ratio_20', 1.0)
    rsi = base_features.get('rsi_14', 50.0)
    spread = base_features.get('spread_bps', 0.0)
    depth_imb = base_features.get('depth_imbalance', 0.0)
    taker = base_features.get('taker_buy_pct_5', 0.5)
    kyle = base_features.get('kyle_lambda_20', 0.0)

    # Interaction: momentum x volume
    momentum_x_volume = ret5 * vol_ratio
    # Interaction: RSI extremes x return direction
    rsi_extreme = 1.0 if rsi > 70 or rsi < 30 else 0.0
    rsi_x_return = (rsi - 50) / 50 * ret1
    # Interaction: spread x depth imbalance
    spread_x_depth = spread * depth_imb
    # Interaction: taker x kyle (liquidity-adjusted flow)
    taker_x_kyle = (taker - 0.5) * 2 * kyle
    # Interaction: vol_ratio x return (breakout signal)
    breakout_signal = vol_ratio * abs(ret1)

    # Lagged features (from Ring Buffer directly)
    c = w['close']; v = w['volume']
    ret_lag_1 = float((c[-2]/c[-3]-1)) if n>=3 else 0.0
    ret_lag_2 = float((c[-3]/c[-4]-1)) if n>=4 else 0.0
    ret_lag_5 = float((c[-6]/c[-7]-1)) if n>=7 else 0.0
    vol_lag_1 = float(v[-2]/(np.mean(v[-22:-2])+1e-10)) if n>=22 else 1.0
    vol_lag_2 = float(v[-3]/(np.mean(v[-23:-3])+1e-10)) if n>=23 else 1.0

    # Momentum acceleration
    mom_accel = ret1 - ret_lag_1
    # Return reversal signal
    reversal = -1.0 * ret_lag_1 * np.sign(ret1) if ret1 * ret_lag_1 < 0 else 0.0

    return {
        'momentum_x_volume': momentum_x_volume,
        'rsi_extreme_flag': rsi_extreme,
        'rsi_x_return': rsi_x_return,
        'spread_x_depth': spread_x_depth,
        'taker_x_kyle': taker_x_kyle,
        'breakout_signal': breakout_signal,
        'return_lag_1': ret_lag_1,
        'return_lag_2': ret_lag_2,
        'return_lag_5': ret_lag_5,
        'volume_ratio_lag_1': vol_lag_1,
        'volume_ratio_lag_2': vol_lag_2,
        'momentum_acceleration': mom_accel,
        'reversal_signal': float(reversal),
        'return_autocorr': float(ret1 * ret_lag_1),
        'vol_return_interaction': float(vol_ratio * ret5),
    }

def _empty_features() -> dict:
    return {k: 0.0 for k in [
        'momentum_x_volume','rsi_extreme_flag','rsi_x_return','spread_x_depth',
        'taker_x_kyle','breakout_signal','return_lag_1','return_lag_2','return_lag_5',
        'volume_ratio_lag_1','volume_ratio_lag_2','momentum_acceleration',
        'reversal_signal','return_autocorr','vol_return_interaction']}
