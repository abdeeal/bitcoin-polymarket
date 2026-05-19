"""
TimescaleDB Queries — All async queries for the 11 dashboard panels.

PRD 13.2: All queries use asyncpg, match exact SQL from PRD spec.
All functions accept a `conn` (asyncpg connection) and return dicts.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ─── Panel 1: Bot Status & Uptime ─────────────────────────────────────────────

async def query_bot_status(conn: Any) -> dict:
    """Panel 1 — Bot state, uptime, last bet, cooling down."""
    uptime_row = await conn.fetchrow("""
        SELECT
            COALESCE(SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))), 0)
                AS total_uptime
        FROM bot_session_log;
    """)

    last_bet_row = await conn.fetchrow("""
        SELECT timestamp, signal, outcome
        FROM bet_log
        ORDER BY timestamp DESC
        LIMIT 1;
    """)

    cd_row = await conn.fetchrow("""
        SELECT started_at, ended_at, duration_sec
        FROM cooling_down_log
        ORDER BY started_at DESC
        LIMIT 1;
    """)

    total_uptime = float(uptime_row['total_uptime'] or 0)
    days = int(total_uptime // 86400)
    hours = int((total_uptime % 86400) // 3600)
    minutes = int((total_uptime % 3600) // 60)

    return {
        'total_uptime_seconds': total_uptime,
        'uptime_formatted': f"{days}d {hours}h {minutes}m",
        'last_bet_timestamp': last_bet_row['timestamp'].isoformat() if last_bet_row else None,
        'last_bet_direction': last_bet_row['signal'] if last_bet_row else None,
        'last_bet_outcome': last_bet_row['outcome'] if last_bet_row else None,
        'last_cooling_down_started_at': cd_row['started_at'].isoformat() if cd_row else None,
        'last_cooling_down_duration_sec': cd_row['duration_sec'] if cd_row else None,
    }


# ─── Panel 2: Balance Overview ────────────────────────────────────────────────

async def query_balance_overview(conn: Any) -> dict:
    """Panel 2 — Starting, current, highest, lowest balance."""
    row = await conn.fetchrow("""
        SELECT
            10.00 AS starting_balance,
            MAX(balance_after)  AS highest_balance,
            MIN(balance_after)  AS lowest_balance,
            (SELECT balance_after FROM bet_log ORDER BY timestamp DESC LIMIT 1)
                AS current_balance
        FROM bet_log
        WHERE outcome IS NOT NULL;
    """)

    if row is None or row['current_balance'] is None:
        return {
            'starting_balance': 10.0,
            'current_balance': 10.0,
            'total_growth_usd': 0.0,
            'total_growth_pct': 0.0,
            'highest_balance': 10.0,
            'lowest_balance': 10.0,
        }

    start = float(row['starting_balance'] or 10.0)
    current = float(row['current_balance'] or 10.0)
    return {
        'starting_balance': start,
        'current_balance': current,
        'total_growth_usd': round(current - start, 2),
        'total_growth_pct': round((current - start) / start * 100, 1) if start > 0 else 0.0,
        'highest_balance': float(row['highest_balance'] or current),
        'lowest_balance': float(row['lowest_balance'] or current),
    }


# ─── Panel 3: Total PnL ───────────────────────────────────────────────────────

async def query_total_pnl(conn: Any, starting_balance: float = 10.0) -> dict:
    """Panel 3 — Gross PnL, fees, net PnL."""
    row = await conn.fetchrow("""
        SELECT
            COALESCE(SUM(pnl), 0)                             AS total_pnl_gross,
            COALESCE(SUM(CASE WHEN outcome THEN
                bet_size * (odds - 1) * 0.02 ELSE 0 END), 0) AS total_fees
        FROM bet_log
        WHERE outcome IS NOT NULL;
    """)
    gross = float(row['total_pnl_gross'] or 0)
    fees = float(row['total_fees'] or 0)
    return {
        'total_pnl_gross': round(gross, 2),
        'total_fees': round(fees, 2),
        'total_pnl_net': round(gross - fees, 2),
        'total_pnl_pct': round(gross / starting_balance * 100, 1) if starting_balance > 0 else 0.0,
    }


# ─── Panel 4: Period Gain/Loss ────────────────────────────────────────────────

async def query_period_gain(conn: Any) -> dict:
    """Panel 4 — Daily, weekly, monthly PnL (Jakarta/UTC+7 day boundaries)."""
    row = await conn.fetchrow("""
        SELECT
            COALESCE(SUM(pnl) FILTER (
                WHERE (timestamp AT TIME ZONE 'Asia/Jakarta')::date
                    = (NOW() AT TIME ZONE 'Asia/Jakarta')::date
            ), 0) AS pnl_today,

            COALESCE(SUM(pnl) FILTER (
                WHERE timestamp AT TIME ZONE 'Asia/Jakarta'
                    >= date_trunc('week', NOW() AT TIME ZONE 'Asia/Jakarta')
            ), 0) AS pnl_week,

            COALESCE(SUM(pnl) FILTER (
                WHERE timestamp AT TIME ZONE 'Asia/Jakarta'
                    >= date_trunc('month', NOW() AT TIME ZONE 'Asia/Jakarta')
            ), 0) AS pnl_month,

            COALESCE(SUM(pnl) FILTER (
                WHERE (timestamp AT TIME ZONE 'Asia/Jakarta')::date
                    = (NOW() AT TIME ZONE 'Asia/Jakarta')::date - INTERVAL '1 day'
            ), 0) AS pnl_yesterday
        FROM bet_log
        WHERE outcome IS NOT NULL;
    """)
    return {
        'pnl_today_usd':     round(float(row['pnl_today']     or 0), 2),
        'pnl_yesterday_usd': round(float(row['pnl_yesterday'] or 0), 2),
        'pnl_week_usd':      round(float(row['pnl_week']      or 0), 2),
        'pnl_month_usd':     round(float(row['pnl_month']     or 0), 2),
    }


# ─── Panel 5: Equity Curve ────────────────────────────────────────────────────

async def query_equity_curve(conn: Any, days: int = 30) -> dict:
    """Panel 5 — Time series of balance per bet, always starting from initial balance."""

    rows = await conn.fetch("""
        SELECT timestamp, balance_after
        FROM bet_log
        WHERE outcome IS NOT NULL
          AND timestamp >= NOW() - ($1 || ' days')::INTERVAL
        ORDER BY timestamp ASC;
    """, str(days))

    cd_rows = await conn.fetch("""
        SELECT started_at, COALESCE(ended_at, NOW()) AS ended_at
        FROM cooling_down_log
        WHERE started_at >= NOW() - ($1 || ' days')::INTERVAL
        ORDER BY started_at;
    """, str(days))

    curve = []
    if rows:
        # User explicitly wants the chart to always begin from $10.00 
        # (the starting capital) as the absolute first node.
        import datetime
        first_ts = rows[0]['timestamp']
        origin_ts = first_ts - datetime.timedelta(minutes=5)
        
        curve.append({
            'timestamp': origin_ts.isoformat(),
            'balance': 10.00,
            'is_start': True,
        })
        # All balance_after points in the selected window
        for r in rows:
            curve.append({
                'timestamp': r['timestamp'].isoformat(),
                'balance': float(r['balance_after']),
                'is_start': False,
            })

    return {
        'equity_curve': curve,
        'cooling_down_periods': [
            {'start': r['started_at'].isoformat(), 'end': r['ended_at'].isoformat()}
            for r in cd_rows
        ],
    }


# ─── Panel 6: Streaks ─────────────────────────────────────────────────────────

async def query_streaks(conn: Any) -> dict:
    """Panel 6 — Trade stats and streak analysis (computed in Python)."""
    rows = await conn.fetch("""
        SELECT outcome, pnl, confidence_tier
        FROM bet_log
        WHERE outcome IS NOT NULL
        ORDER BY timestamp ASC;
    """)
    if not rows:
        return _empty_streaks()

    outcomes = np.array([bool(r['outcome']) for r in rows], dtype=bool)
    pnls = np.array([float(r['pnl']) for r in rows])
    tiers = np.array([str(r['confidence_tier'] or '') for r in rows])

    return compute_streaks(outcomes, pnls, tiers)


def compute_streaks(outcomes: np.ndarray, pnls: np.ndarray, tiers: np.ndarray = None) -> dict:
    """Vectorized streak computation (PRD 13.3 Panel 6)."""
    arr = outcomes.astype(np.int8)
    changes = np.diff(arr, prepend=arr[0] + 1)
    groups = np.split(arr, np.where(changes != 0)[0])
    streaks = [(bool(g[0]), len(g)) for g in groups if len(g) > 0]

    win_streaks = [l for v, l in streaks if v]
    loss_streaks = [l for v, l in streaks if not v]
    current = streaks[-1] if streaks else (True, 0)

    wins = outcomes.sum()
    losses = (~outcomes).sum()
    total = len(outcomes)
    win_pnls = pnls[outcomes]
    loss_pnls = pnls[~outcomes]

    gross_wins = float(win_pnls.sum()) if len(win_pnls) > 0 else 0.0
    gross_losses = float(abs(loss_pnls.sum())) if len(loss_pnls) > 0 else 0.0

    high_conf_wr = 0.0
    if tiers is not None:
        high_conf_mask = np.char.find(tiers, 'HIGH') >= 0
        hc_outcomes = outcomes[high_conf_mask]
        if len(hc_outcomes) > 0:
            high_conf_wr = float(hc_outcomes.sum() / len(hc_outcomes))

    return {
        'total_entries': int(total),
        'total_wins': int(wins),
        'total_losses': int(losses),
        'win_rate': round(float(wins / max(total, 1)), 4),
        'max_win_streak': max(win_streaks, default=0),
        'max_lose_streak': max(loss_streaks, default=0),
        'current_win_streak': current[1] if current[0] else 0,
        'current_lose_streak': current[1] if not current[0] else 0,
        'avg_win_usd': round(float(win_pnls.mean()), 4) if len(win_pnls) > 0 else 0.0,
        'avg_loss_usd': round(float(loss_pnls.mean()), 4) if len(loss_pnls) > 0 else 0.0,
        'profit_factor': round(gross_wins / max(gross_losses, 1e-10), 2),
        'largest_win_usd': round(float(win_pnls.max()), 4) if len(win_pnls) > 0 else 0.0,
        'largest_loss_usd': round(float(loss_pnls.min()), 4) if len(loss_pnls) > 0 else 0.0,
        'high_conf_win_rate': round(high_conf_wr, 4),
    }


def _empty_streaks() -> dict:
    return {k: 0 for k in [
        'total_entries', 'total_wins', 'total_losses', 'win_rate',
        'max_win_streak', 'max_lose_streak', 'current_win_streak',
        'current_lose_streak', 'avg_win_usd', 'avg_loss_usd',
        'profit_factor', 'largest_win_usd', 'largest_loss_usd', 'high_conf_win_rate'
    ]}


# ─── Panel 7 & 8: Performance by Side ────────────────────────────────────────

async def query_by_side(conn: Any) -> list[dict]:
    """Panels 7 & 8 — PnL and win rate by signal direction."""
    rows = await conn.fetch("""
        SELECT
            CASE WHEN signal LIKE 'BET_UP%' THEN 'BUY' ELSE 'SELL' END AS side,
            COUNT(*)                                AS entries,
            COUNT(*) FILTER (WHERE outcome = true)  AS wins,
            SUM(pnl)                                AS total_pnl,
            AVG(pnl)                                AS avg_pnl,
            ROUND(COUNT(*) FILTER (WHERE outcome = true)::numeric
                  / NULLIF(COUNT(*), 0) * 100, 1)  AS win_rate_pct
        FROM bet_log
        WHERE outcome IS NOT NULL
        GROUP BY side
        ORDER BY side;
    """)
    return [
        {
            'side': r['side'],
            'entries': int(r['entries']),
            'wins': int(r['wins']),
            'total_pnl': round(float(r['total_pnl'] or 0), 2),
            'avg_pnl': round(float(r['avg_pnl'] or 0), 4),
            'win_rate_pct': float(r['win_rate_pct'] or 0),
        }
        for r in rows
    ]


# ─── Panel 9: By Hour ─────────────────────────────────────────────────────────

async def query_by_hour(conn: Any) -> list[dict]:
    """Panel 9 — Performance by Jakarta (UTC+7) hour."""
    rows = await conn.fetch("""
        SELECT
            EXTRACT(HOUR FROM timestamp AT TIME ZONE 'Asia/Jakarta') AS hour,
            COUNT(*)                                AS total_bets,
            COALESCE(SUM(pnl), 0)                   AS total_pnl,
            COALESCE(AVG(pnl), 0)                   AS avg_pnl,
            ROUND(COALESCE(AVG(outcome::int), 0) * 100, 1) AS win_rate_pct
        FROM bet_log
        WHERE outcome IS NOT NULL
        GROUP BY hour
        ORDER BY hour;
    """)
    data_dict = {int(r['hour']): r for r in rows}
    full_data = []
    for h in range(24):
        if h in data_dict:
            r = data_dict[h]
            full_data.append({
                'hour': h,
                'total_bets': int(r['total_bets']),
                'total_pnl': round(float(r['total_pnl']), 2),
                'avg_pnl': round(float(r['avg_pnl']), 4),
                'win_rate_pct': float(r['win_rate_pct']),
            })
        else:
            full_data.append({
                'hour': h,
                'total_bets': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate_pct': 0.0,
            })
    return full_data


# ─── Panel 10: By Day ─────────────────────────────────────────────────────────

async def query_by_day(conn: Any) -> list[dict]:
    """Panel 10 — Performance by day of week (Jakarta/UTC+7)."""
    rows = await conn.fetch("""
        SELECT
            TO_CHAR(timestamp AT TIME ZONE 'Asia/Jakarta', 'Dy') AS day_name,
            EXTRACT(DOW FROM timestamp AT TIME ZONE 'Asia/Jakarta') AS day_num,
            COUNT(*)                                     AS total_bets,
            COALESCE(SUM(pnl), 0)                        AS total_pnl,
            ROUND(COALESCE(AVG(outcome::int), 0) * 100, 1) AS win_rate_pct
        FROM bet_log
        WHERE outcome IS NOT NULL
        GROUP BY day_name, day_num
        ORDER BY day_num;
    """)
    days_map = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    data_dict = {int(r['day_num']): r for r in rows}
    
    # Monday to Sunday order
    day_order = [1, 2, 3, 4, 5, 6, 0]
    
    full_data = []
    for d in day_order:
        if d in data_dict:
            r = data_dict[d]
            full_data.append({
                'day_name': r['day_name'],
                'day_num': d,
                'total_bets': int(r['total_bets']),
                'total_pnl': round(float(r['total_pnl']), 2),
                'win_rate_pct': float(r['win_rate_pct']),
            })
        else:
            full_data.append({
                'day_name': days_map[d],
                'day_num': d,
                'total_bets': 0,
                'total_pnl': 0.0,
                'win_rate_pct': 0.0,
            })
    return full_data


# ─── Panel 11: Calendar ───────────────────────────────────────────────────────

async def query_calendar(conn: Any, year: int, month: int) -> list[dict]:
    """Panel 11 — Daily PnL for calendar view (Jakarta/UTC+7)."""
    rows = await conn.fetch("""
        SELECT
            DATE(timestamp AT TIME ZONE 'Asia/Jakarta') AS trade_date,
            COUNT(*)                                    AS total_bets,
            COALESCE(SUM(pnl), 0)                       AS total_pnl,
            ROUND(COALESCE(AVG(outcome::int), 0) * 100, 1) AS win_rate_pct,
            COALESCE(SUM(pnl), 0) > 0                  AS is_profit_day
        FROM bet_log
        WHERE outcome IS NOT NULL
          AND EXTRACT(YEAR  FROM timestamp AT TIME ZONE 'Asia/Jakarta') = $1
          AND EXTRACT(MONTH FROM timestamp AT TIME ZONE 'Asia/Jakarta') = $2
        GROUP BY trade_date
        ORDER BY trade_date;
    """, year, month)
    return [
        {
            'date': r['trade_date'].isoformat(),
            'total_bets': int(r['total_bets']),
            'total_pnl': round(float(r['total_pnl']), 2),
            'win_rate_pct': float(r['win_rate_pct']),
            'is_profit_day': bool(r['is_profit_day']),
        }
        for r in rows
    ]


# ─── Latest Prediction ────────────────────────────────────────────────────────

async def query_latest_prediction(conn: Any) -> dict:
    """Query the latest model prediction for the Next Prediction card."""
    import math
    import time

    # Compute next 5-min candle window
    now = time.time()
    interval = 300
    next_ts = math.ceil(now / interval) * interval
    remaining = max(0, int(next_ts - now))
    target_time = time.strftime('%H:%M', time.localtime(next_ts))

    pred_row = await conn.fetchrow("""
        SELECT timestamp, model_prob, signal, confidence_tier, bot_state
        FROM model_predictions
        ORDER BY timestamp DESC
        LIMIT 1;
    """)

    if pred_row is None:
        return {
            'direction': 'SKIP',
            'target_time': target_time,
            'confidence': 0.0,
            'remaining_seconds': remaining,
            'reason': 'Waiting for first candle',
        }

    signal = pred_row['signal'] or 'NO_BET'
    prob = float(pred_row['model_prob'] or 0.5)

    # Map signal to direction
    # Confidence = how certain the model is about any direction
    # For SKIP: max(prob, 1-prob) shows the conviction level even when below threshold
    if 'UP' in signal:
        direction = 'BUY'
        confidence = prob
    elif 'DOWN' in signal:
        direction = 'SELL'
        confidence = 1.0 - prob
    else:
        direction = 'SKIP'
        confidence = max(prob, 1.0 - prob)

    # Determine reason
    tier = pred_row['confidence_tier'] or ''
    if direction == 'SKIP':
        reason = f'Prob {prob:.1%} - below threshold (need >72% or <28%)'
    elif 'HIGH' in tier:
        reason = f'High confidence - {prob:.1%}'
    else:
        reason = f'Normal signal - {prob:.1%}'

    return {
        'direction': direction,
        'target_time': target_time,
        'confidence': round(confidence, 4),
        'raw_prob': round(prob, 4),
        'remaining_seconds': remaining,
        'reason': reason,
    }



# --- Trade History (History Page) ---

async def query_trade_history(
    conn,
    date_from=None,
    date_to=None,
):
    """
    Return all resolved bets ordered newest-first.
    Optional date_from / date_to filters (YYYY-MM-DD, inclusive, UTC+7).
    """
    conditions = ["outcome IS NOT NULL"]
    params = []

    if date_from:
        params.append(date_from)
        conditions.append(f"(timestamp AT TIME ZONE 'Asia/Jakarta')::date >= ${len(params)}::date")

    if date_to:
        params.append(date_to)
        conditions.append(f"(timestamp AT TIME ZONE 'Asia/Jakarta')::date <= ${len(params)}::date")

    where = " AND ".join(conditions)
    rows = await conn.fetch(f"""
        SELECT
            id,
            timestamp,
            signal,
            confidence_tier,
            model_prob,
            bet_size,
            odds,
            outcome,
            pnl,
            balance_before,
            balance_after,
            bracket_index
        FROM bet_log
        WHERE {where}
        ORDER BY timestamp DESC;
    """, *params)

    result = []
    for i, r in enumerate(rows):
        ts = r["timestamp"]
        result.append({
            "id":              str(r.get("id")) if r.get("id") else str(i),
            "timestamp":       ts.isoformat() if ts else None,
            "signal":          r["signal"],
            "confidence_tier": r["confidence_tier"],
            "model_prob":      float(r["model_prob"] or 0),
            "bet_size":        float(r["bet_size"]   or 0),
            "odds":            float(r["odds"]       or 0),
            "outcome":         bool(r["outcome"]),
            "pnl":             round(float(r["pnl"]  or 0), 4),
            "balance_before":  round(float(r["balance_before"] or 0), 2),
            "balance_after":   round(float(r["balance_after"]  or 0), 2),
            "bracket_index":   int(r["bracket_index"] or 0),
        })
    return result
