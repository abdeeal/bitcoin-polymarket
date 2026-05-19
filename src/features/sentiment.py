"""
Sentiment & Derivatives Features — 10 features, fully NumPy vectorized.
Target latency: <10ms.
"""
import numpy as np

def compute_sentiment(w: dict) -> dict:
    n = len(w['close'])
    if n < 5: return _empty_features()
    funding = w.get('funding', np.zeros(n))
    oi = w.get('oi', np.zeros(n))
    liq_l = w.get('liq_long', np.zeros(n))
    liq_s = w.get('liq_short', np.zeros(n))
    iv = w.get('iv_25delta', np.zeros(n))
    pcr = w.get('put_call_ratio', np.zeros(n))

    fr = float(funding[-1])
    fr_ma5 = float(np.mean(funding[-5:])) if n>=5 else fr
    oi_val = float(oi[-1])
    oi_chg = float((oi[-1]-oi[-2])/(oi[-2]+1e-10)) if n>=2 and oi[-2]>0 else 0.0
    oi_chg5 = float((oi[-1]-oi[-5])/(oi[-5]+1e-10)) if n>=5 and oi[-5]>0 else 0.0
    liq_ratio = float(liq_l[-1]/(liq_s[-1]+1e-10)) if liq_s[-1]>0 else 1.0
    liq_total = float(liq_l[-1]+liq_s[-1])
    iv_val = float(iv[-1])
    pcr_val = float(pcr[-1])
    fear_greed = 0.5 + (fr*100) + (oi_chg*10) - (iv_val/200)

    return {
        'funding_rate': fr, 'funding_rate_ma5': fr_ma5,
        'oi_current': oi_val, 'oi_change_1': oi_chg, 'oi_change_5': oi_chg5,
        'liq_ratio': liq_ratio, 'liq_total_usd': liq_total,
        'iv_25delta': iv_val, 'put_call_ratio': pcr_val,
        'fear_greed_proxy': float(np.clip(fear_greed, 0, 1)),
    }

def _empty_features() -> dict:
    return {k: 0.0 for k in [
        'funding_rate','funding_rate_ma5','oi_current','oi_change_1','oi_change_5',
        'liq_ratio','liq_total_usd','iv_25delta','put_call_ratio','fear_greed_proxy']}
