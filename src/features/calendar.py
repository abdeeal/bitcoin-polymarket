"""
Calendar/Time Features — 8 features, pure Python (trivial).
Target latency: <1ms.
"""
import math
from datetime import datetime, timezone

def compute_calendar() -> dict:
    now = datetime.now(timezone.utc)
    hour = now.hour; dow = now.weekday(); minute = now.minute
    hour_sin = math.sin(2*math.pi*hour/24)
    hour_cos = math.cos(2*math.pi*hour/24)
    dow_sin = math.sin(2*math.pi*dow/7)
    dow_cos = math.cos(2*math.pi*dow/7)
    is_weekend = 1.0 if dow >= 5 else 0.0
    # Asian=0-8 UTC, Europe=8-16, US=16-24
    session = 0 if hour<8 else (1 if hour<16 else 2)
    is_high_vol_hour = 1.0 if hour in (13,14,15,16,20,21) else 0.0
    return {
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'dow_sin': dow_sin, 'dow_cos': dow_cos,
        'is_weekend': is_weekend, 'session': float(session),
        'is_high_vol_hour': is_high_vol_hour,
        'minute_of_hour': float(minute),
    }
