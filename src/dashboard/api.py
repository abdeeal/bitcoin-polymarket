"""
FastAPI Dashboard API — 9 REST endpoints + 1 WebSocket.

PRD 13.2 Endpoints:
  GET  /api/stats/summary       → panels 1–3
  GET  /api/stats/period        → panel 4
  GET  /api/stats/equity-curve  → panel 5
  GET  /api/stats/streaks       → panel 6
  GET  /api/stats/by-side       → panels 7–8
  GET  /api/stats/by-hour       → panel 9
  GET  /api/stats/by-day        → panel 10
  GET  /api/stats/by-calendar   → panel 11
  GET  /api/system/status       → bot state, cooling down
  GET  /api/system/features-health → all 150 feature health checks
  WS   /ws/live                 → real-time updates every 5s

Startup: asyncpg connection pool created on lifespan.
All DB calls are async (non-blocking event loop).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog
import httpx

logger = structlog.get_logger(__name__)

# ── Lazy imports so the module loads without DB being available ───────────────
try:
    import asyncpg  # type: ignore
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from src.dashboard import queries as q
from src.dashboard.ws_broadcaster import WSBroadcaster

# Global state — populated on startup
_pool: Optional[Any] = None
_broadcaster: Optional[WSBroadcaster] = None

# Build DATABASE_URL from env vars (same as main.py)
_DB_HOST = os.getenv("TIMESCALE_HOST", "localhost")
_DB_PORT = os.getenv("TIMESCALE_PORT", "5432")
_DB_NAME = os.getenv("TIMESCALE_DB", "btc_prediction")
_DB_USER = os.getenv("TIMESCALE_USER", "btcbot")
_DB_PASS = os.getenv("TIMESCALE_PASSWORD", "")
DATABASE_URL = f"postgresql://{_DB_USER}:{_DB_PASS}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB pool on startup, close on shutdown."""
    global _pool, _broadcaster

    if HAS_ASYNCPG:
        try:
            _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            logger.info("db_pool_created")
        except Exception as e:
            logger.warning("db_pool_failed", error=str(e), fallback="stub mode")
            _pool = None
    else:
        logger.warning("asyncpg_not_installed", fallback="stub mode")
        _pool = None

    _broadcaster = WSBroadcaster(pool=_pool)

    yield  # app running

    if _pool:
        await _pool.close()
    logger.info("db_pool_closed")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BTC Prediction Bot Dashboard",
    version="1.0.0",
    description="Real-time monitoring dashboard for the BTC prediction bot.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

async def _get_conn():
    if _pool is None:
        return None
    return await _pool.acquire()


async def _release(conn):
    if _pool and conn:
        await _pool.release(conn)


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/stats/summary")
async def get_summary():
    """Panels 1–3: bot status, balance overview, total PnL."""
    conn = await _get_conn()
    try:
        if conn is None:
            return _stub_summary()
        status = await q.query_bot_status(conn)
        balance = await q.query_balance_overview(conn)
        pnl = await q.query_total_pnl(conn, balance.get('starting_balance', 10.0))
        return {'status': status, 'balance': balance, 'pnl': pnl}
    finally:
        await _release(conn)


@app.get("/api/stats/period")
async def get_period_gain():
    """Panel 4: daily, weekly, monthly PnL."""
    conn = await _get_conn()
    try:
        if conn is None:
            return _stub_period()
        return await q.query_period_gain(conn)
    finally:
        await _release(conn)


@app.get("/api/stats/equity-curve")
async def get_equity_curve(days: int = Query(default=30, ge=1, le=365)):
    """Panel 5: time-series balance + cooling down periods."""
    conn = await _get_conn()
    try:
        if conn is None:
            return _stub_equity()
        return await q.query_equity_curve(conn, days=days)
    finally:
        await _release(conn)


@app.get("/api/stats/streaks")
async def get_streaks():
    """Panel 6: trade statistics and streak analysis."""
    conn = await _get_conn()
    try:
        if conn is None:
            return q._empty_streaks()
        return await q.query_streaks(conn)
    finally:
        await _release(conn)


@app.get("/api/stats/by-side")
async def get_by_side():
    """Panels 7–8: performance and win rate by direction."""
    conn = await _get_conn()
    try:
        if conn is None:
            return []
        return await q.query_by_side(conn)
    finally:
        await _release(conn)


@app.get("/api/stats/by-hour")
async def get_by_hour():
    """Panel 9: performance by Jakarta (UTC+7) hour."""
    conn = await _get_conn()
    try:
        if conn is None:
            return []
        return await q.query_by_hour(conn)
    finally:
        await _release(conn)


@app.get("/api/trades/history")
async def get_trade_history(
    date_from: Optional[str] = Query(default=None, description="Start date YYYY-MM-DD"),
    date_to:   Optional[str] = Query(default=None, description="End date YYYY-MM-DD"),
):
    """Full trade log for the History page (paginated on frontend)."""
    conn = await _get_conn()
    try:
        if conn is None:
            return []
        return await q.query_trade_history(conn, date_from=date_from, date_to=date_to)
    finally:
        await _release(conn)


@app.get("/api/stats/by-day")
async def get_by_day():
    """Panel 10: performance by day of week."""
    conn = await _get_conn()
    try:
        if conn is None:
            return []
        return await q.query_by_day(conn)
    finally:
        await _release(conn)


@app.get("/api/stats/by-calendar")
async def get_calendar(
    year: int = Query(default=2026, ge=2024, le=2030),
    month: int = Query(default=1, ge=1, le=12),
):
    """Panel 11: daily performance for calendar view."""
    conn = await _get_conn()
    try:
        if conn is None:
            return []
        return await q.query_calendar(conn, year=year, month=month)
    finally:
        await _release(conn)


@app.get("/api/system/status")
async def get_system_status():
    """Bot state, cooling down status, circuit breaker."""
    conn = await _get_conn()
    try:
        if conn is None:
            return _stub_system_status()
        return await q.query_bot_status(conn)
    finally:
        await _release(conn)


@app.get("/api/prediction/latest")
async def get_latest_prediction():
    """Latest model prediction for the Next Prediction card."""
    conn = await _get_conn()
    try:
        if conn is None:
            return _stub_prediction()
        return await q.query_latest_prediction(conn)
    finally:
        await _release(conn)

@app.get("/api/trades/detail/{bet_id}")
async def get_trade_detail(bet_id: str):
    """Per-model confidence and feature attribution for a specific trade."""
    conn = await _get_conn()
    try:
        if conn is None:
            return {"error": "Database not connected"}
        import json as _json
        row = await conn.fetchrow("""
            SELECT
                bd.prob_lgbm,
                bd.prob_xgb,
                bd.prob_lstm,
                bd.prob_ensemble,
                bd.features_pro,
                bd.features_contra,
                bl.signal,
                bl.model_prob,
                bl.confidence_tier,
                bl.outcome,
                bl.pnl
            FROM bet_details bd
            JOIN bet_log bl ON bl.id = bd.bet_id
            WHERE bd.bet_id = $1::uuid;
        """, bet_id)

        if not row:
            return {"error": "No details found for this trade", "bet_id": bet_id}

        # Parse JSONB columns
        pro = row['features_pro']
        contra = row['features_contra']
        if isinstance(pro, str):
            pro = _json.loads(pro)
        if isinstance(contra, str):
            contra = _json.loads(contra)

        return {
            "bet_id": bet_id,
            "signal": row['signal'],
            "model_prob": float(row['model_prob'] or 0),
            "confidence_tier": row['confidence_tier'],
            "outcome": bool(row['outcome']) if row['outcome'] is not None else None,
            "pnl": float(row['pnl'] or 0),
            "prob_ensemble": float(row['prob_ensemble'] or 0),
            "prob_lgbm": float(row['prob_lgbm']) if row['prob_lgbm'] is not None else None,
            "prob_xgb": float(row['prob_xgb']) if row['prob_xgb'] is not None else None,
            "prob_lstm": float(row['prob_lstm']) if row['prob_lstm'] is not None else None,
            "features_pro": pro or [],
            "features_contra": contra or [],
        }
    finally:
        await _release(conn)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time push every 5 seconds — all panels refreshed."""
    await websocket.accept()
    logger.info("ws_client_connected", client=websocket.client)

    if _broadcaster is None:
        logger.warning("ws_no_broadcaster", reason="broadcaster not initialized")
        await websocket.close(code=1011, reason="Server not ready")
        return

    try:
        await _broadcaster.handle_client(websocket)
    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", client=websocket.client)
    except Exception as e:
        logger.error("ws_unexpected_error", error=str(e), client=websocket.client)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ── Health & Features Monitoring ──────────────────────────────────────────────

# Complete feature registry: (feature_name, category, data_source)
# data_source is one of: "binance_rest", "binance_ws", "coinglass", "deribit", "polymarket", "internal"
_FEATURE_REGISTRY = []

# Price Action — 20 features (Binance REST Klines)
_PA_FEATURES = [
    'return_1m', 'return_5m', 'return_10m', 'return_20m', 'return_60m',
    'log_return_1m', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
    'realized_vol_5', 'realized_vol_20', 'realized_vol_60',
    'price_position_20', 'price_position_60',
    'dist_from_high_20', 'dist_from_low_20',
    'consecutive_green', 'consecutive_red', 'range_pct', 'gap',
]
for f in _PA_FEATURES:
    _FEATURE_REGISTRY.append((f, "Price Action", "binance_rest"))

# Volume — 15 features (Binance REST + WS)
_VOL_FEATURES = [
    'volume_ratio_5', 'volume_ratio_20', 'volume_ratio_60',
    'taker_bs_ratio', 'taker_buy_pct_5', 'taker_buy_pct_20',
    'volume_trend_20', 'volume_std_20', 'volume_cv_20',
    'obv_delta_5', 'cvd_5', 'cvd_20', 'vp_divergence',
    'large_trade_ratio', 'avg_trade_size',
]
for f in _VOL_FEATURES:
    _FEATURE_REGISTRY.append((f, "Volume", "binance_rest"))

# Technical — 35 features (Binance REST via TA-Lib)
_TECH_FEATURES = [
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd_line', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
    'atr_7', 'atr_14', 'atr_ratio',
    'ema_5', 'ema_10', 'ema_20', 'ema_50',
    'ema_5_20_diff', 'ema_10_50_diff', 'price_vs_ema_20',
    'stoch_k_14', 'stoch_d_14',
    'cci_14', 'williams_r_14',
    'adx_14', 'plus_di', 'minus_di',
    'mfi_14', 'obv_slope_5', 'sar_signal',
    'rsi_momentum', 'macd_cross', 'bb_squeeze', 'atr_normalized', 'trend_strength',
]
for f in _TECH_FEATURES:
    _FEATURE_REGISTRY.append((f, "Technical", "binance_rest"))

# Order Book — 20 features (Binance WS Depth)
_OB_FEATURES = [
    'depth_imbalance', 'depth_imbalance_5m_avg',
    'ofi', 'ofi_rolling_5', 'ofi_rolling_20',
    'spread_bps', 'spread_mean_20', 'spread_zscore',
    'bid_ask_ratio_5', 'bid_ask_ratio_10',
    'depth_1pct_ratio', 'bid_depth_change', 'ask_depth_change',
    'depth_disparity_5', 'total_depth', 'total_depth_change',
    'bid_depth', 'ask_depth', 'mid_price', 'spread_raw',
]
for f in _OB_FEATURES:
    _FEATURE_REGISTRY.append((f, "Order Book", "binance_ws"))

# Microstructure — 15 features (Binance REST + WS)
_MICRO_FEATURES = [
    'kyle_lambda_10', 'kyle_lambda_20', 'kyle_lambda_60', 'kyle_prank',
    'amihud_10', 'amihud_20', 'amihud_60',
    'vwap_dev_20', 'vwap_dev_60',
    'vp_correlation_20',
    'taker_ratio', 'taker_mean_5', 'taker_mean_20', 'taker_momentum',
    'trade_count',
]
for f in _MICRO_FEATURES:
    _FEATURE_REGISTRY.append((f, "Microstructure", "binance_rest"))

# Cross-Asset — 12 features (Binance REST multi-symbol)
_CROSS_FEATURES = [
    'btc_eth_corr_20', 'btc_sol_corr_20',
    'btc_vs_eth_5', 'btc_vs_sol_5',
    'eth_btc_ratio', 'eth_btc_ratio_change',
    'vol_regime_zscore', 'trend_regime',
    'mean_reversion_zscore',
    'btc_dominance_proxy', 'cross_asset_momentum', 'btc_ret_5',
]
for f in _CROSS_FEATURES:
    _FEATURE_REGISTRY.append((f, "Cross-Asset", "binance_rest"))

# Sentiment & Derivatives — 10 features (Coinglass + Deribit)
_SENT_FEATURES_CG = [
    'funding_rate', 'funding_rate_ma5',
    'oi_current', 'oi_change_1', 'oi_change_5',
    'liq_ratio', 'liq_total_usd',
    'fear_greed_proxy',
]
for f in _SENT_FEATURES_CG:
    _FEATURE_REGISTRY.append((f, "Sentiment", "coinglass"))

_SENT_FEATURES_DERIBIT = [
    'iv_25delta', 'put_call_ratio',
]
for f in _SENT_FEATURES_DERIBIT:
    _FEATURE_REGISTRY.append((f, "Sentiment", "deribit"))

# Calendar — 8 features (Internal computation)
_CAL_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'session', 'is_high_vol_hour', 'minute_of_hour',
]
for f in _CAL_FEATURES:
    _FEATURE_REGISTRY.append((f, "Calendar", "internal"))

# Lagged & Interaction — 15 features (Internal computation)
_LAG_FEATURES = [
    'momentum_x_volume', 'rsi_extreme_flag', 'rsi_x_return',
    'spread_x_depth', 'taker_x_kyle', 'breakout_signal',
    'return_lag_1', 'return_lag_2', 'return_lag_5',
    'volume_ratio_lag_1', 'volume_ratio_lag_2',
    'momentum_acceleration', 'reversal_signal',
    'return_autocorr', 'vol_return_interaction',
]
for f in _LAG_FEATURES:
    _FEATURE_REGISTRY.append((f, "Lagged & Interaction", "internal"))

# Source → API details
_SOURCE_META = {
    "binance_rest": {
        "api_url": "https://fapi.binance.com",
        "ping_url": "https://fapi.binance.com/fapi/v1/ping",
        "env_key": "BINANCE_API_KEY",
        "env_placeholder": None,  # Binance public endpoints don't require API key
    },
    "binance_ws": {
        "api_url": "wss://fstream.binance.com",
        "ping_url": "https://fapi.binance.com/fapi/v1/ping",  # REST ping as proxy
        "env_key": "BINANCE_API_KEY",
        "env_placeholder": None,
    },
    "coinglass": {
        "api_url": "https://open-api.coinglass.com",
        "ping_url": "https://open-api.coinglass.com/public/v2/index/fear-greed-history",
        "env_key": "COINGLASS_API_KEY",
        "env_placeholder": "your_coinglass_api_key_here",
    },
    "deribit": {
        "api_url": "https://www.deribit.com/api/v2",
        "ping_url": "https://www.deribit.com/api/v2/public/test",
        "env_key": "DERIBIT_CLIENT_ID",
        "env_placeholder": "your_deribit_client_id_here",
    },
    "polymarket": {
        "api_url": "https://clob.polymarket.com",
        "ping_url": "https://clob.polymarket.com/time",
        "env_key": "POLYMARKET_API_KEY",
        "env_placeholder": "your_polymarket_api_key_here",
    },
    "internal": {
        "api_url": "Internal Engine",
        "ping_url": None,
        "env_key": None,
        "env_placeholder": None,
    },
}


@app.get("/api/system/features-health")
async def get_features_health():
    """
    Real-time health check for ALL 150 features.
    
    Each feature is listed individually with:
      - feature name
      - category (Price Action, Volume, Technical, etc.)
      - data source API URL
      - status (Online / Offline / Not Configured)
      - note (HTTP status or error reason)
    """
    # Step 1: Check each unique data source once
    source_status = {}
    
    for source_key, meta in _SOURCE_META.items():
        if source_key == "internal":
            source_status["internal"] = {"status": "Online", "note": "200 OK"}
            continue
            
        # Check if API key is configured
        env_key = meta["env_key"]
        env_placeholder = meta["env_placeholder"]
        
        if env_placeholder is not None:
            # This source requires an API key
            current_val = os.getenv(env_key, "")
            if not current_val or current_val == env_placeholder:
                source_status[source_key] = {
                    "status": "Not Configured",
                    "note": f"API key not set"
                }
                continue
        
        # API key is set (or not required) — try pinging
        ping_url = meta["ping_url"]
        if not ping_url:
            source_status[source_key] = {"status": "Online", "note": "200 OK"}
            continue
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(ping_url)
                if resp.status_code == 200:
                    source_status[source_key] = {"status": "Online", "note": f"200 OK"}
                else:
                    source_status[source_key] = {
                        "status": "Offline",
                        "note": f"HTTP {resp.status_code} — {resp.reason_phrase}"
                    }
        except httpx.ConnectTimeout:
            source_status[source_key] = {"status": "Offline", "note": "Connection timed out (5s)"}
        except httpx.ConnectError as e:
            source_status[source_key] = {"status": "Offline", "note": f"Connection refused — {str(e)[:80]}"}
        except Exception as e:
            source_status[source_key] = {"status": "Offline", "note": str(e)[:100]}
    
    # Step 2: Build per-feature result from registry
    features = []
    for idx, (feature_name, category, source_key) in enumerate(_FEATURE_REGISTRY):
        meta = _SOURCE_META[source_key]
        src = source_status.get(source_key, {"status": "Unknown", "note": "—"})
        features.append({
            "id": idx + 1,
            "feature": feature_name,
            "category": category,
            "source": source_key,
            "api_url": meta["api_url"],
            "status": src["status"],
            "note": src["note"],
        })
    
    # Step 3: Build summary counts
    online = sum(1 for f in features if f["status"] == "Online")
    offline = sum(1 for f in features if f["status"] == "Offline")
    not_configured = sum(1 for f in features if f["status"] == "Not Configured")
    
    return {
        "total": len(features),
        "online": online,
        "offline": offline,
        "not_configured": not_configured,
        "sources": {k: v for k, v in source_status.items()},
        "features": features,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "db_connected": _pool is not None}


# ── Stub Data (no DB) ─────────────────────────────────────────────────────────

def _stub_summary() -> dict:
    return {
        'status': {
            'bot_state': 'BOOT',
            'total_uptime_seconds': 0,
            'uptime_formatted': '0m',
            'last_bet_timestamp': None,
            'last_bet_direction': None,
            'last_bet_outcome': None,
            'last_cooling_down_started_at': None,
            'last_cooling_down_duration_sec': None,
        },
        'balance': {
            'starting_balance': 10.0,
            'current_balance': 10.0,
            'total_growth_usd': 0.0,
            'total_growth_pct': 0.0,
            'highest_balance': 10.0,
            'lowest_balance': 10.0,
        },
        'pnl': {
            'total_pnl_gross': 0.0,
            'total_fees': 0.0,
            'total_pnl_net': 0.0,
            'total_pnl_pct': 0.0,
        },
    }


def _stub_period() -> dict:
    return {
        'pnl_today_usd': 0.0,
        'pnl_yesterday_usd': 0.0,
        'pnl_week_usd': 0.0,
        'pnl_month_usd': 0.0,
    }


def _stub_equity() -> dict:
    return {'equity_curve': [], 'cooling_down_periods': []}


def _stub_system_status() -> dict:
    return {
        'bot_state': 'BOOT',
        'cooling_down_active': False,
        'cooling_down_remaining_seconds': 0,
    }


def _stub_prediction() -> dict:
    import math
    import time as _time
    now = _time.time()
    interval = 300
    next_ts = math.ceil(now / interval) * interval
    return {
        'direction': 'SKIP',
        'target_time': _time.strftime('%H:%M', _time.localtime(next_ts)),
        'confidence': 0.0,
        'remaining_seconds': max(0, int(next_ts - now)),
        'reason': 'Waiting for data',
    }
