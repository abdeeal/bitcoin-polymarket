"""
WebSocket Broadcaster — Real-time push every 5 seconds to all connected clients.

PRD 13.4: Pushes a payload containing bot state, balance, streaks,
cooling down status, last bet, and today's PnL.

Maintains a set of active WebSocket connections.
Each client connection handled independently.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Optional, Set

import structlog

logger = structlog.get_logger(__name__)

BROADCAST_INTERVAL = 5.0   # seconds


class WSBroadcaster:
    """
    Broadcasts real-time updates to all WebSocket clients every 5 seconds.

    Each client connects and receives continuous JSON pushes.
    Data sourced from DB pool (async) or stubbed if unavailable.
    """

    def __init__(self, pool: Optional[Any] = None):
        self._pool = pool
        self._active_clients: Set = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._start_time: float = time.time()
        # Cache last successful prediction to avoid flashing SKIP on DB errors
        self._last_prediction: Optional[dict] = None
        self._last_payload: Optional[dict] = None

    async def handle_client(self, websocket) -> None:
        """
        Handle a single WebSocket client connection.

        Uses two concurrent tasks:
        1. Sender — pushes payload every 5 seconds
        2. Receiver — listens for client messages / disconnect

        This pattern ensures we properly detect when the client disconnects
        (browser close, navigation, etc.) instead of only discovering it
        on the next failed send_text(), which could take up to 5 seconds
        and silently kill the loop.
        """
        self._active_clients.add(websocket)
        logger.info("ws_client_added", total=len(self._active_clients))

        sender_task = asyncio.create_task(self._sender_loop(websocket))
        receiver_task = asyncio.create_task(self._receiver_loop(websocket))

        try:
            # Wait for either task to finish (whichever exits first)
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel the other task
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            self._active_clients.discard(websocket)
            logger.info("ws_client_removed", total=len(self._active_clients))

    async def _sender_loop(self, websocket) -> None:
        """Push payload every BROADCAST_INTERVAL seconds."""
        try:
            while True:
                payload = await self._build_payload()
                await websocket.send_text(json.dumps(payload))
                await asyncio.sleep(BROADCAST_INTERVAL)
        except Exception as e:
            logger.debug("ws_sender_ended", reason=str(e))

    async def _receiver_loop(self, websocket) -> None:
        """
        Listen for incoming messages from the client.

        In Starlette/FastAPI, calling receive_text() is what actually
        detects a WebSocket disconnect. Without this, the server only
        discovers the client is gone when send_text() fails, which can
        cause a stale connection to linger.

        We also support a 'ping' → 'pong' keep-alive from the client.
        """
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == 'ping':
                    await websocket.send_text('pong')
        except Exception as e:
            logger.debug("ws_receiver_ended", reason=str(e))

    async def _build_payload(self) -> dict:
        """Build the real-time update payload (PRD 13.4)."""
        conn = None
        try:
            if self._pool is not None:
                conn = await self._pool.acquire()
                payload = await self._payload_from_db(conn)
                # Cache successful payload and prediction
                self._last_payload = payload
                self._last_prediction = payload.get('next_prediction')
                return payload
            else:
                return self._stub_payload()
        except Exception as e:
            logger.warning("broadcaster_payload_error", error=str(e))
            # Return cached payload if available, otherwise stub
            if self._last_payload:
                # Update countdown in cached payload
                candle = self._next_candle_info()
                cached = dict(self._last_payload)
                cached['ts'] = time.time()
                cached['uptime_formatted'] = self._format_uptime(time.time() - self._start_time)
                cached['uptime_seconds'] = time.time() - self._start_time
                if 'next_prediction' in cached:
                    cached['next_prediction'] = dict(cached['next_prediction'])
                    cached['next_prediction']['remaining_seconds'] = candle['remaining_seconds']
                    cached['next_prediction']['target_time'] = candle['target_time']
                return cached
            return self._stub_payload()
        finally:
            if self._pool and conn:
                await self._pool.release(conn)

    def _format_uptime(self, seconds: float) -> str:
        """Format seconds into human-readable uptime string."""
        s = int(seconds)
        d = s // 86400
        h = (s % 86400) // 3600
        m = (s % 3600) // 60
        if d > 0:
            return f"{d}d {h}h {m}m"
        elif h > 0:
            return f"{h}h {m}m"
        else:
            return f"{m}m"

    @staticmethod
    def _next_candle_info() -> dict:
        """Compute the next 5-minute candle window and remaining seconds."""
        now = time.time()
        interval = 300  # 5 minutes
        next_ts = math.ceil(now / interval) * interval
        remaining = max(0, int(next_ts - now))
        target_time = time.strftime('%H:%M', time.localtime(next_ts))
        return {
            'next_ts': next_ts,
            'remaining_seconds': remaining,
            'target_time': target_time,
        }

    async def _query_next_prediction(self, conn) -> dict:
        """
        Query the latest model prediction from the database.

        Falls back to a 'Waiting for data' stub if no predictions exist yet.
        """
        candle = self._next_candle_info()

        # Get the most recent prediction from model_predictions table
        pred_row = await conn.fetchrow("""
            SELECT timestamp, model_prob, signal, confidence_tier, bot_state
            FROM model_predictions
            ORDER BY timestamp DESC
            LIMIT 1;
        """)

        if pred_row is None:
            # No predictions yet — bot hasn't run a cycle
            return {
                'direction': 'SKIP',
                'target_time': candle['target_time'],
                'confidence': 0.0,
                'remaining_seconds': candle['remaining_seconds'],
                'reason': 'Waiting for first candle',
            }

        signal = pred_row['signal'] or 'NO_BET'
        prob = float(pred_row['model_prob'] or 0.5)

        # Map signal to direction
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
            reason = f'Below threshold'
        elif 'HIGH' in tier:
            reason = f'High confidence - {prob:.1%}'
        else:
            reason = f'Normal signal - {prob:.1%}'

        return {
            'direction': direction,
            'target_time': candle['target_time'],
            'confidence': round(confidence, 4),
            'raw_prob': round(prob, 4),
            'remaining_seconds': candle['remaining_seconds'],
            'reason': reason,
        }

    def _next_prediction_stub(self) -> dict:
        """
        Fallback prediction when DB is not available.
        Shows correct next-interval timestamp with waiting status.
        """
        candle = self._next_candle_info()
        return {
            'direction': 'SKIP',
            'target_time': candle['target_time'],
            'confidence': 0.0,
            'remaining_seconds': candle['remaining_seconds'],
            'reason': 'Waiting for data',
        }

    async def _payload_from_db(self, conn) -> dict:
        """Query DB for real-time fields."""
        from src.dashboard.queries import query_streaks, query_period_gain

        period = await query_period_gain(conn)
        streaks = await query_streaks(conn)

        last_bet_row = await conn.fetchrow("""
            SELECT timestamp, signal, outcome, balance_after
            FROM bet_log ORDER BY timestamp DESC LIMIT 1;
        """)

        portfolio_row = await conn.fetchrow("""
            SELECT balance, consecutive_losses FROM portfolio_state WHERE id = 1;
        """)

        cd_row = await conn.fetchrow("""
            SELECT started_at, EXTRACT(EPOCH FROM (NOW() - started_at)) as elapsed 
            FROM cooling_down_log
            WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1;
        """)

        current_balance = float(portfolio_row['balance']) if portfolio_row else 10.0
        consecutive_losses = int(portfolio_row['consecutive_losses']) if portfolio_row else 0

        cooling_down_active = False
        cd_remaining = 0
        if cd_row:
            elapsed = float(cd_row['elapsed'] or 0)
            if elapsed < 1800:
                cooling_down_active = True
                cd_remaining = 1800 - int(elapsed)

        uptime_seconds = time.time() - self._start_time

        # Query latest prediction from DB
        next_prediction = await self._query_next_prediction(conn)

        return {
            'ts': time.time(),
            'bot_state': 'COOLING_DOWN' if cooling_down_active else 'ACTIVE',
            'current_balance': current_balance,
            'consecutive_losses': consecutive_losses,
            'cooling_down_active': cooling_down_active,
            'cooling_down_remaining_seconds': cd_remaining,
            'uptime_formatted': self._format_uptime(uptime_seconds),
            'uptime_seconds': uptime_seconds,
            'last_bet': {
                'timestamp': last_bet_row['timestamp'].isoformat() if last_bet_row else None,
                'direction': last_bet_row['signal'] if last_bet_row else None,
                'outcome': bool(last_bet_row['outcome']) if last_bet_row else None,
                'balance_after': float(last_bet_row['balance_after']) if last_bet_row else None,
            },
            'today_pnl': period.get('pnl_today_usd', 0.0),
            'current_win_streak': streaks.get('current_win_streak', 0),
            'current_lose_streak': streaks.get('current_lose_streak', 0),
            'win_rate': streaks.get('win_rate', 0.0),
            'next_prediction': next_prediction,
        }

    def _stub_payload(self) -> dict:
        uptime_seconds = time.time() - self._start_time
        return {
            'ts': time.time(),
            'bot_state': 'BOOT',
            'current_balance': 10.0,
            'consecutive_losses': 0,
            'cooling_down_active': False,
            'cooling_down_remaining_seconds': 0,
            'uptime_formatted': self._format_uptime(uptime_seconds),
            'uptime_seconds': uptime_seconds,
            'last_bet': {'timestamp': None, 'direction': None, 'outcome': None, 'balance_after': None},
            'today_pnl': 0.0,
            'current_win_streak': 0,
            'current_lose_streak': 0,
            'win_rate': 0.0,
            'next_prediction': self._next_prediction_stub(),
        }

    @property
    def client_count(self) -> int:
        return len(self._active_clients)
