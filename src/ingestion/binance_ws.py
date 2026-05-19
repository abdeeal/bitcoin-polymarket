"""
Binance WebSocket Client — I/O layer data ingestion.

Architecture Rules:
  - This module runs in the asyncio event loop (I/O thread)
  - NEVER imports numpy, pandas, or ML code
  - Writes raw ticks to Ring Buffer and async persistence queue
  - Handles reconnection with exponential backoff
  - Combines multiple streams into a single WebSocket connection

Streams:
  - btcusdt@kline_1m     → OHLCV 1-minute candles
  - btcusdt@depth20@100ms → Order book L2 (20 levels)
  - btcusdt@aggTrade      → Aggregated trades
  - btcusdt@bookTicker    → Best bid/ask
  - ethusdt@kline_1m      → ETH cross-asset
  - solusdt@kline_1m      → SOL cross-asset
"""

import asyncio
import json
import time
from typing import Optional, Callable, Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Binance combined stream base URL
BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"

# Individual stream names
STREAMS = [
    "btcusdt@kline_1m",
    "btcusdt@depth20@100ms",
    "btcusdt@aggTrade",
    "btcusdt@bookTicker",
    "ethusdt@kline_1m",
    "solusdt@kline_1m",
]

# Max reconnect delay (exponential backoff caps here)
MAX_RECONNECT_DELAY = 60  # seconds
INITIAL_RECONNECT_DELAY = 1  # seconds


class BinanceWebSocketClient:
    """
    Async WebSocket client for Binance market data streams.

    Uses Binance's combined stream endpoint to multiplex multiple
    streams over a single WebSocket connection (stays within the
    5-connection-per-IP limit).

    All incoming messages are parsed and dispatched to registered
    handlers, which write to the Ring Buffer.
    """

    def __init__(
        self,
        on_kline: Optional[Callable] = None,
        on_depth: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_book_ticker: Optional[Callable] = None,
        on_cross_asset_kline: Optional[Callable] = None,
        persistence_queue: Optional[asyncio.Queue] = None,
    ):
        """
        Initialize the WebSocket client with message handlers.

        Args:
            on_kline: Callback for BTC 1m kline events. Signature: async (bar: dict) -> None
            on_depth: Callback for order book depth events.
            on_trade: Callback for aggregated trade events.
            on_book_ticker: Callback for best bid/ask updates.
            on_cross_asset_kline: Callback for ETH/SOL kline events.
            persistence_queue: Async queue for fire-and-forget DB writes.
        """
        self._on_kline = on_kline
        self._on_depth = on_depth
        self._on_trade = on_trade
        self._on_book_ticker = on_book_ticker
        self._on_cross_asset_kline = on_cross_asset_kline
        self._persistence_queue = persistence_queue

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running: bool = False
        self._reconnect_delay: float = INITIAL_RECONNECT_DELAY

        # Statistics
        self._messages_received: int = 0
        self._last_message_time: float = 0
        self._connect_count: int = 0
        self._error_count: int = 0

    async def connect(self) -> None:
        """
        Connect to Binance combined stream WebSocket.

        Uses a single connection with all streams multiplexed.
        Automatically retries on failure with exponential backoff.
        """
        self._running = True
        stream_param = "/".join(STREAMS)
        url = f"{BINANCE_WS_BASE}?streams={stream_param}"

        logger.info("binance_ws_connecting", url=url, streams=len(STREAMS))

        while self._running:
            try:
                self._session = aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(
                    url,
                    heartbeat=20,  # ping every 20s to keep alive
                    receive_timeout=30,
                )
                self._connect_count += 1
                self._reconnect_delay = INITIAL_RECONNECT_DELAY

                logger.info(
                    "binance_ws_connected",
                    connect_count=self._connect_count,
                )

                await self._message_loop()

            except (
                aiohttp.WSServerHandshakeError,
                aiohttp.ClientConnectionError,
                ConnectionResetError,
                asyncio.TimeoutError,
            ) as e:
                self._error_count += 1
                logger.warning(
                    "binance_ws_connection_error",
                    error=str(e),
                    reconnect_delay=self._reconnect_delay,
                    error_count=self._error_count,
                )
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "binance_ws_unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                await self._cleanup()

            if self._running:
                logger.info(
                    "binance_ws_reconnecting",
                    delay=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, MAX_RECONNECT_DELAY
                )

    async def _message_loop(self) -> None:
        """
        Main message receiving loop.

        Processes each message and dispatches to the appropriate handler.
        Never performs CPU-heavy work — only parsing and dispatching.
        """
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._messages_received += 1
                self._last_message_time = time.time()

                try:
                    data = json.loads(msg.data)
                    stream = data.get('stream', '')
                    payload = data.get('data', {})
                    await self._dispatch(stream, payload)
                except json.JSONDecodeError:
                    logger.warning("binance_ws_invalid_json", data=msg.data[:200])
                except Exception as e:
                    logger.error(
                        "binance_ws_handler_error",
                        error=str(e),
                        stream=data.get('stream', 'unknown'),
                    )

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                logger.warning(
                    "binance_ws_closed",
                    type=str(msg.type),
                    extra=str(msg.extra) if hasattr(msg, 'extra') else None,
                )
                break

    async def _dispatch(self, stream: str, payload: dict) -> None:
        """
        Dispatch a parsed message to the appropriate handler.

        This is the routing layer — each stream type maps to a specific
        handler that transforms raw data into Ring Buffer format.
        """
        if stream == "btcusdt@kline_1m":
            await self._handle_btc_kline(payload)
        elif stream == "btcusdt@depth20@100ms":
            await self._handle_depth(payload)
        elif stream == "btcusdt@aggTrade":
            await self._handle_agg_trade(payload)
        elif stream == "btcusdt@bookTicker":
            await self._handle_book_ticker(payload)
        elif stream in ("ethusdt@kline_1m", "solusdt@kline_1m"):
            await self._handle_cross_asset_kline(stream, payload)

    async def _handle_btc_kline(self, payload: dict) -> None:
        """
        Process BTC 1-minute kline event.

        Only processes closed candles (k.x == True) to ensure
        we're working with finalized data.

        Binance kline payload structure:
          k.t = open time (ms)
          k.o/h/l/c = OHLC prices
          k.v = total volume
          k.q = quote asset volume
          k.V = taker buy base volume
          k.Q = taker buy quote volume
          k.n = number of trades
          k.x = is this candle closed?
        """
        k = payload.get('k', {})
        is_closed = k.get('x', False)

        if not is_closed:
            return  # Only process completed candles

        bar = {
            'timestamp': int(k.get('t', 0)),
            'open':      float(k.get('o', 0)),
            'high':      float(k.get('h', 0)),
            'low':       float(k.get('l', 0)),
            'close':     float(k.get('c', 0)),
            'volume':    float(k.get('v', 0)),
            'taker_buy': float(k.get('V', 0)),
            'trade_count': float(k.get('n', 0)),
        }

        # Dispatch to Ring Buffer writer
        if self._on_kline:
            await self._on_kline(bar)

        # Fire-and-forget to persistence queue
        if self._persistence_queue:
            try:
                self._persistence_queue.put_nowait(('candle', bar))
            except asyncio.QueueFull:
                logger.warning("persistence_queue_full", data_type="candle")

        logger.debug(
            "btc_kline_processed",
            timestamp=bar['timestamp'],
            close=bar['close'],
            volume=bar['volume'],
        )

    async def _handle_depth(self, payload: dict) -> None:
        """
        Process order book depth snapshot (L2, 20 levels).

        Computes aggregated metrics from raw bid/ask levels:
          - Total bid/ask depth
          - Spread
          - Depth within 1% of mid price
          - Bid/ask ratios at 5 and 10 levels

        Binance depth20 payload:
          bids: [[price, qty], ...] (20 levels)
          asks: [[price, qty], ...] (20 levels)
        """
        bids = payload.get('bids', [])
        asks = payload.get('asks', [])

        if not bids or not asks:
            return

        # Best bid/ask
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # Total depth (all 20 levels)
        bid_depth = sum(float(b[1]) for b in bids)
        ask_depth = sum(float(a[1]) for a in asks)

        # Depth within 1% of mid price
        threshold_1pct = mid * 0.01
        bid_depth_1pct = sum(
            float(b[1]) for b in bids
            if mid - float(b[0]) <= threshold_1pct
        )
        ask_depth_1pct = sum(
            float(a[1]) for a in asks
            if float(a[0]) - mid <= threshold_1pct
        )

        # Bid/ask ratios at different level counts
        bid_qty_5 = sum(float(b[1]) for b in bids[:5])
        ask_qty_5 = sum(float(a[1]) for a in asks[:5])
        bid_qty_10 = sum(float(b[1]) for b in bids[:10])
        ask_qty_10 = sum(float(a[1]) for a in asks[:10])

        depth_data = {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'spread': spread,
            'mid_price': mid,
            'bid_depth_1pct': bid_depth_1pct,
            'ask_depth_1pct': ask_depth_1pct,
            'bid_ask_ratio_5': bid_qty_5 / (ask_qty_5 + 1e-10),
            'bid_ask_ratio_10': bid_qty_10 / (ask_qty_10 + 1e-10),
        }

        if self._on_depth:
            await self._on_depth(depth_data)

        # Persist snapshot
        if self._persistence_queue:
            try:
                self._persistence_queue.put_nowait(('orderbook', {
                    'timestamp': int(time.time() * 1000),
                    'bids': bids[:10],  # Top 10 for storage
                    'asks': asks[:10],
                    **depth_data,
                }))
            except asyncio.QueueFull:
                pass  # Non-critical persistence

    async def _handle_agg_trade(self, payload: dict) -> None:
        """
        Process aggregated trade event.

        Binance aggTrade payload:
          p = price
          q = quantity
          m = buyer is maker (True = sell/taker sell, False = buy/taker buy)
          T = trade time (ms)
        """
        trade = {
            'timestamp': int(payload.get('T', 0)),
            'price': float(payload.get('p', 0)),
            'quantity': float(payload.get('q', 0)),
            'is_buyer_maker': payload.get('m', False),
        }

        if self._on_trade:
            await self._on_trade(trade)

    async def _handle_book_ticker(self, payload: dict) -> None:
        """
        Process best bid/ask ticker update.

        Binance bookTicker payload:
          b = best bid price
          B = best bid qty
          a = best ask price
          A = best ask qty
        """
        ticker = {
            'best_bid': float(payload.get('b', 0)),
            'best_bid_qty': float(payload.get('B', 0)),
            'best_ask': float(payload.get('a', 0)),
            'best_ask_qty': float(payload.get('A', 0)),
        }

        if self._on_book_ticker:
            await self._on_book_ticker(ticker)

    async def _handle_cross_asset_kline(self, stream: str, payload: dict) -> None:
        """
        Process ETH or SOL 1-minute kline event.

        Only processes closed candles, same as BTC handler.
        """
        k = payload.get('k', {})
        is_closed = k.get('x', False)

        if not is_closed:
            return

        symbol = 'eth' if 'ethusdt' in stream else 'sol'
        bar = {
            'timestamp': int(k.get('t', 0)),
            'symbol': symbol,
            'close': float(k.get('c', 0)),
            'volume': float(k.get('v', 0)),
        }

        if self._on_cross_asset_kline:
            await self._on_cross_asset_kline(bar)

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        await self._cleanup()
        logger.info(
            "binance_ws_disconnected",
            total_messages=self._messages_received,
            total_connects=self._connect_count,
            total_errors=self._error_count,
        )

    async def _cleanup(self) -> None:
        """Clean up WebSocket and HTTP session resources."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._ws = None
        self._session = None

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is currently connected."""
        return self._ws is not None and not self._ws.closed

    def stats(self) -> dict:
        """Return connection statistics for monitoring."""
        return {
            'is_connected': self.is_connected,
            'messages_received': self._messages_received,
            'last_message_time': self._last_message_time,
            'connect_count': self._connect_count,
            'error_count': self._error_count,
            'reconnect_delay': self._reconnect_delay,
        }
