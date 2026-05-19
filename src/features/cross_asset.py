"""
Cross-Asset & Regime Features — 12 features, fully NumPy vectorized.
Target latency: <10ms.
"""
import numpy as np
from src.features.vectorization_standards import compute_returns, rolling_std_np
from src.features.jit_kernels import compute_rolling_correlation

def compute_cross_asset(w: dict) -> dict:
    c = w['close']; n = len(c)
    if n < 10: return _empty_features()
    eth = w.get('eth_close', np.zeros(n)); sol = w.get('sol_close', np.zeros(n))

    btc_eth_corr = btc_sol_corr = 0.0
    if n >= 20:
        br = np.diff(c)/(c[:-1]+1e-10)
        if np.any(eth>0):
            er = np.diff(eth)/(eth[:-1]+1e-10)
            cr = compute_rolling_correlation(br, er, 20)
            btc_eth_corr = float(cr[-1]) if not np.isnan(cr[-1]) else 0.0
        if np.any(sol>0):
            sr = np.diff(sol)/(sol[:-1]+1e-10)
            cr = compute_rolling_correlation(br, sr, 20)
            btc_sol_corr = float(cr[-1]) if not np.isnan(cr[-1]) else 0.0

    br5 = (c[-1]/c[-6]-1) if n>=6 else 0.0
    er5 = (eth[-1]/eth[-6]-1) if n>=6 and eth[-6]>0 else 0.0
    sr5 = (sol[-1]/sol[-6]-1) if n>=6 and sol[-6]>0 else 0.0

    eth_btc = float(eth[-1]/c[-1]) if eth[-1]>0 and c[-1]>0 else 0.0
    eth_btc_chg = 0.0
    if n>=6 and c[-6]>0 and eth[-6]>0:
        prev = eth[-6]/c[-6]; eth_btc_chg = float((eth_btc-prev)/(prev+1e-10))

    vol_regime = trend_regime = mean_rev = 0.0
    if n >= 60:
        ret = compute_returns(c); v20 = rolling_std_np(ret, 20)
        vv = v20[~np.isnan(v20)]
        if len(vv)>=20:
            m,s = float(np.mean(vv[-60:])), float(np.std(vv[-60:]))
            if s>1e-10: vol_regime = float((vv[-1]-m)/s)
    if n >= 20:
        e20 = np.convolve(c, np.ones(20)/20, mode='valid')
        if len(e20)>=5: trend_regime = float(np.sign(e20[-1]-e20[-5]))
        m20,s20 = float(np.mean(c[-20:])), float(np.std(c[-20:]))
        if s20>1e-10: mean_rev = float((c[-1]-m20)/s20)

    return {
        'btc_eth_corr_20': btc_eth_corr, 'btc_sol_corr_20': btc_sol_corr,
        'btc_vs_eth_5': float(br5-er5), 'btc_vs_sol_5': float(br5-sr5),
        'eth_btc_ratio': eth_btc, 'eth_btc_ratio_change': eth_btc_chg,
        'vol_regime_zscore': vol_regime, 'trend_regime': trend_regime,
        'mean_reversion_zscore': mean_rev,
        'btc_dominance_proxy': float(br5-(er5+sr5)/2),
        'cross_asset_momentum': float((er5+sr5)/2), 'btc_ret_5': float(br5),
    }

def _empty_features() -> dict:
    return {k: 0.0 for k in [
        'btc_eth_corr_20','btc_sol_corr_20','btc_vs_eth_5','btc_vs_sol_5',
        'eth_btc_ratio','eth_btc_ratio_change','vol_regime_zscore','trend_regime',
        'mean_reversion_zscore','btc_dominance_proxy','cross_asset_momentum','btc_ret_5']}
