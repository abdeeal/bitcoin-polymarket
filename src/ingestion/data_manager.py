"""
Data Manager — Orchestrates all data ingestion with I/O/CPU separation.

Architecture Rules:
  - This is the TOP-LEVEL orchestrator for the I/O layer
  - Manages WebSocket client, REST client, and derivatives polling
  - Routes all incoming data to the Ring Buffer
  - Pushes all data to the persistence queue (fire-and-forget)
  - Handles trade aggregation (aggTrade → per-candle microstructure)
  - NEVER performs numpy/pandas/ML computation

Key Responsibilities:
  1. Start/stop all data streams
  2. Aggregate real-time trades into per-candle microstructure metrics
  3. Merge order book snapshots with candle data in Ring Buffer
  4. Coordinate REST backfill on cold boot
  5. Detect data gaps and trigger REST catch-up
"""

import asyncio
import time
from typing import Optional
from collections import defaultdict

import structlog

from src.memory.ring_buffer import RingBuffer
from src.ingestion.binance_ws import BinanceWebSocketClient
from src.ingestion.binance_rest import BinanceRESTClient
from src.ingestion.derivatives_client import DerivativesClient

logger = structlog.get_logger(__name__)

# Trade aggregation window (ms) — aggregate trades per 1-minute candle
TRADE_AGG_WINDOW_MS = 60_000


class TradeAggregator:
    """
    Aggregates real-time aggTrade events into per-candle microstructure metrics.

    Tracks within the current 1-minute window:
      - Total trade count
      - Large trade count (> threshold)
      - Average trade size
      - Taker buy ratio
      - Total buy/sell volume

    Resets at each candle boundary.
    """

    LARGE_TRADE_THRESHOLD_BTC = 0.5  # 0.5 BTC = "large" trade

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators for a new candle window."""
        self._trade_count: int = 0
        self._large_trade_count: int = 0
        self._total_qty: float = 0.0
        self._buy_qty: float = 0.0
        self._sell_qty: float = 0.0
        self._window_start: int = 0

    def add_trade(self, trade: dict) -> None:
        """
        Add a single trade to the current aggregation window.

        Args:
            trade: Dict with 'price', 'quantity', 'is_buyer_maker', 'timestamp'.
        """
        qty = trade.get('quantity', 0)
        is_sell = trade.get('is_buyer_maker', False)

        self._trade_count += 1
        self._total_qty += qty

        if qty >= self.LARGE_TRADE_THRESHOLD_BTC:
            self._large_trade_count += 1

        if is_sell:
            self._sell_qty += qty
        else:
            self._buy_qty += qty

    def snapshot(self) -> dict:
        """
        Return current aggregated metrics and reset for next window.

        Returns:
            Dict with microstructure metrics for the current candle.
        """
        total = self._buy_qty + self._sell_qty
        metrics = {
            'trade_count': float(self._trade_count),
            'large_trade_count': float(self._large_trade_count),
            'avg_trade_size': self._total_qty / max(self._trade_count, 1),
            'taker_buy_ratio': self._buy_qty / max(total, 1e-10),
        }
        self.reset()
        return metrics


class DataManager:
    """
    Top-level orchestrator for all data ingestion.

    Manages the lifecycle of all data source connections and
    coordinates data flow into the Ring Buffer and persistence queue.

    Architecture:
      - I/O operations run in the asyncio event loop
      - Ring Buffer writes are the only cross-thread operation
      - Persistence queue writes are non-blocking (fire-and-forget)
      - CPU layer (feature engine) reads from Ring Buffer independently
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        persistence_queue: asyncio.Queue,
        binance_api_key: Optional[str] = None,
        binance_api_secret: Optional[str] = None,
        deribit_client_id: Optional[str] = None,
        deribit_client_secret: Optional[str] = None,
        coinglass_api_key: Optional[str] = None,
    ):
        """
        Initialize the data manager with all sub-clients.

        Args:
            ring_buffer: Shared Ring Buffer for in-memory data.
            persistence_queue: Async queue for DB writes.
            binance_api_key: Binance API key.
            binance_api_secret: Binance API secret.
            deribit_client_id: Deribit client ID.
            deribit_client_secret: Deribit client secret.
            coinglass_api_key: Coinglass API key.
        """
        self._ring_buffer = ring_buffer
        self._persistence_queue = persistence_queue

        # Trade aggregator for microstructure features
        self._trade_aggregator = TradeAggregator()

        # Latest order book snapshot (updated in real-time)
        self._latest_depth: dict = {}

        # Latest cross-asset prices
        self._latest_eth: dict = {}
        self._latest_sol: dict = {}

        # WebSocket client with handlers bound to this manager
        self._ws_client = BinanceWebSocketClient(
            on_kline=self._on_btc_kline,
            on_depth=self._on_depth,
            on_trade=self._on_trade,
            on_book_ticker=self._on_book_ticker,
            on_cross_asset_kline=self._on_cross_asset_kline,
            persistence_queue=persistence_queue,
        )

        # REST client for historical backfill
        self._rest_client = BinanceRESTClient(
            api_key=binance_api_key,
            api_secret=binance_api_secret,
        )

        # Derivatives polling client
        self._derivatives_client = DerivativesClient(
            ring_buffer=ring_buffer,
            persistence_queue=persistence_queue,
            deribit_client_id=deribit_client_id,
            deribit_client_secret=deribit_client_secret,
            coinglass_api_key=coinglass_api_key,
        )

        # Running tasks
        self._tasks: list[asyncio.Task] = []
        self._running: bool = False

        # Data gap detection
        self._last_candle_time: int = 0
        self._gap_count: int = 0

    # =========================================================================
    # WebSocket Event Handlers (I/O Layer)
    # =========================================================================

    async def _on_btc_kline(self, bar: dict) -> None:
        """
        Handle a completed BTC 1-minute candle.

        This is the main data entry point. When a candle closes:
          1. Merge microstructure metrics from trade aggregator
          2. Merge latest order book snapshot
          3. Merge latest cross-asset prices
          4. Push the enriched bar to Ring Buffer
          5. Check for data gaps

        Args:
            bar: OHLCV bar dict from WebSocket handler.
        """
        # 1. Merge microstructure metrics
        micro = self._trade_aggregator.snapshot()
        bar.update(micro)

        # 2. Merge latest order book data
        if self._latest_depth:
            bar.update(self._latest_depth)

        # 3. Merge latest cross-asset prices
        if self._latest_eth:
            bar['eth_close'] = self._latest_eth.get('close', 0)
            bar['eth_volume'] = self._latest_eth.get('volume', 0)
        if self._latest_sol:
            bar['sol_close'] = self._latest_sol.get('close', 0)
            bar['sol_volume'] = self._latest_sol.get('volume', 0)

        # 4. Push to Ring Buffer
        self._ring_buffer.push(bar)

        # 5. Check for data gaps
        self._check_gap(bar.get('timestamp', 0))

        logger.debug(
            "candle_ingested",
            timestamp=bar.get('timestamp'),
            close=bar.get('close'),
            volume=bar.get('volume'),
            buffer_size=self._ring_buffer.size,
        )

    async def _on_depth(self, depth_data: dict) -> None:
        """
        Handle order book depth update.

        Stores the latest snapshot for merging with the next candle.
        Also updates the current (latest) Ring Buffer entry with
        real-time order book data.
        """
        self._latest_depth = depth_data
        # Update the latest bar in Ring Buffer with fresh OB data
        self._ring_buffer.update_latest(depth_data)

    async def _on_trade(self, trade: dict) -> None:
        """
        Handle aggregated trade event.

        Feeds the trade aggregator, which accumulates microstructure
        metrics within each 1-minute window.
        """
        self._trade_aggregator.add_trade(trade)

    async def _on_book_ticker(self, ticker: dict) -> None:
        """
        Handle best bid/ask ticker update.

        Updates spread in the latest depth snapshot.
        """
        if ticker.get('best_bid') and ticker.get('best_ask'):
            spread = ticker['best_ask'] - ticker['best_bid']
            mid = (ticker['best_bid'] + ticker['best_ask']) / 2
            self._latest_depth.update({
                'spread': spread,
                'mid_price': mid,
            })

    async def _on_cross_asset_kline(self, bar: dict) -> None:
        """
        Handle ETH or SOL kline event.

        Stores latest cross-asset price for merging with next BTC candle.
        """
        symbol = bar.get('symbol', '')
        if symbol == 'eth':
            self._latest_eth = bar
        elif symbol == 'sol':
            self._latest_sol = bar

    # =========================================================================
    # Data Gap Detection
    # =========================================================================

    def _check_gap(self, timestamp: int) -> None:
        """
        Detect gaps in candle data (>2 consecutive candles missing).

        If a gap is detected, logs a warning and increments gap counter.
        The boot sequence can use this to trigger REST catch-up.
        """
        if self._last_candle_time > 0:
            expected_gap = 60_000  # 1 minute in ms
            actual_gap = timestamp - self._last_candle_time
            missed = (actual_gap // expected_gap) - 1

            if missed > 0:
                self._gap_count += int(missed)
                logger.warning(
                    "data_gap_detected",
                    missed_candles=missed,
                    last_timestamp=self._last_candle_time,
                    current_timestamp=timestamp,
                    total_gaps=self._gap_count,
                )

                if missed >= 2:
                    # TODO: Trigger REST catch-up for missed candles
                    logger.error(
                        "critical_data_gap",
                        missed_candles=missed,
                        action="rest_catchup_needed",
                    )

        self._last_candle_time = timestamp

    # =========================================================================
    # Backfill (Cold Boot)
    # =========================================================================

    async def backfill_historical(self, n_candles: int = 600) -> None:
        """
        Perform full historical backfill and populate Ring Buffer.

        Called during cold boot. Fetches BTC, ETH, SOL candles and
        derivatives data, then writes everything to the Ring Buffer.

        Args:
            n_candles: Number of 1-minute candles to backfill.
        """
        t0 = time.perf_counter()
        logger.info("data_manager_backfill_start", n_candles=n_candles)

        # Fetch all historical data
        backfill_data = await self._rest_client.full_backfill(n_candles)

        btc_candles = backfill_data.get('btc_candles', [])
        eth_candles = backfill_data.get('eth_candles', [])
        sol_candles = backfill_data.get('sol_candles', [])
        funding = backfill_data.get('funding', [])
        oi_history = backfill_data.get('oi_history', [])

        # Build ETH/SOL lookup by timestamp for efficient merging
        eth_by_ts = {c['timestamp']: c for c in eth_candles}
        sol_by_ts = {c['timestamp']: c for c in sol_candles}

        # Build funding/OI lookup (nearest match by timestamp)
        funding_by_ts = {f['timestamp']: f['funding'] for f in funding}
        oi_by_ts = {o['timestamp']: o['oi'] for o in oi_history}

        # Merge all data into enriched bars
        enriched_bars = []
        for btc_bar in btc_candles:
            ts = btc_bar['timestamp']

            # Merge cross-asset data (exact timestamp match)
            eth = eth_by_ts.get(ts, {})
            sol = sol_by_ts.get(ts, {})
            btc_bar['eth_close'] = eth.get('eth_close', 0)
            btc_bar['eth_volume'] = eth.get('eth_volume', 0)
            btc_bar['sol_close'] = sol.get('sol_close', 0)
            btc_bar['sol_volume'] = sol.get('sol_volume', 0)

            # Merge funding (nearest available)
            nearest_funding = self._find_nearest(funding_by_ts, ts)
            btc_bar['funding'] = nearest_funding if nearest_funding else 0

            # Merge OI (nearest available)
            nearest_oi = self._find_nearest(oi_by_ts, ts)
            btc_bar['oi'] = nearest_oi if nearest_oi else 0

            enriched_bars.append(btc_bar)

        # Batch-push to Ring Buffer (single lock acquisition)
        self._ring_buffer.push_batch(enriched_bars)

        # Also push to persistence queue for DB storage
        for bar in enriched_bars:
            try:
                self._persistence_queue.put_nowait(('candle', bar))
            except asyncio.QueueFull:
                break  # Stop if queue is full — backfill data is already in RB

        elapsed = time.perf_counter() - t0
        logger.info(
            "data_manager_backfill_complete",
            elapsed_sec=round(elapsed, 2),
            bars_loaded=len(enriched_bars),
            buffer_size=self._ring_buffer.size,
        )

    @staticmethod
    def _find_nearest(lookup: dict, target_ts: int) -> Optional[float]:
        """Find the value with the nearest timestamp key."""
        if not lookup:
            return None
        # Find the key closest to target_ts
        nearest_key = min(lookup.keys(), key=lambda k: abs(k - target_ts))
        # Only use if within 1 hour
        if abs(nearest_key - target_ts) <= 3_600_000:
            return lookup[nearest_key]
        return None

    # =========================================================================
    # Stream Lifecycle
    # =========================================================================

    async def connect_all_streams(self) -> None:
        """
        Start all live data streams.

        Launches WebSocket connection and derivatives polling as
        concurrent asyncio tasks.
        """
        self._running = True

        # Start WebSocket connection (runs indefinitely with auto-reconnect)
        ws_task = asyncio.create_task(
            self._ws_client.connect(),
            name="binance_ws",
        )
        self._tasks.append(ws_task)

        # Start derivatives polling
        deriv_task = asyncio.create_task(
            self._derivatives_client.start_polling(),
            name="derivatives_polling",
        )
        self._tasks.append(deriv_task)

        logger.info(
            "data_manager_streams_started",
            tasks=len(self._tasks),
        )

    async def disconnect_all(self) -> None:
        """Stop all data streams and clean up."""
        self._running = False

        await self._ws_client.disconnect()
        await self._derivatives_client.stop_polling()
        await self._rest_client.close()

        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()
        logger.info("data_manager_streams_stopped")

    # =========================================================================
    # Data Freshness Check
    # =========================================================================

    async def check_data_freshness(self) -> bool:
        """
        Check if existing Ring Buffer data is fresh enough for warm start.

        Returns True if the buffer has data and the latest candle is
        less than 1 hour old. Otherwise returns False (cold start needed).
        """
        latest_ts = self._ring_buffer.latest_timestamp
        if latest_ts is None:
            return False

        now_ms = int(time.time() * 1000)
        age_ms = now_ms - latest_ts
        is_fresh = age_ms < 3_600_000  # 1 hour

        logger.info(
            "data_freshness_check",
            latest_timestamp=latest_ts,
            age_seconds=round(age_ms / 1000),
            is_fresh=is_fresh,
            buffer_size=self._ring_buffer.size,
        )

        return is_fresh

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> dict:
        """Return comprehensive data manager statistics."""
        return {
            'running': self._running,
            'buffer': self._ring_buffer.stats(),
            'websocket': self._ws_client.stats(),
            'rest': self._rest_client.stats(),
            'derivatives': self._derivatives_client.stats(),
            'gap_count': self._gap_count,
            'active_tasks': len([t for t in self._tasks if not t.done()]),
        }
