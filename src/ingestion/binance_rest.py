"""
Binance REST Client — Historical data backfill for cold boot.

Architecture Rules:
  - This module runs in the asyncio event loop (I/O thread)
  - Fetches historical candles, funding rates, and OI via REST
  - Writes directly to Ring Buffer AND pushes to persistence queue
  - Used only during boot sequence — never during live trading
  - Target: 600 candles backfill completes in <90 seconds

Endpoints:
  - GET /api/v3/klines           → Historical OHLCV (spot)
  - GET /fapi/v1/fundingRate     → Funding rate history
  - GET /fapi/v1/openInterest    → Open interest history
"""

import asyncio
import time
from typing import Optional
from datetime import datetime, timezone

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Binance API base URLs
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Rate limiting: Binance allows 1200 requests/minute for spot
# We use conservative pacing to avoid hitting limits during backfill
REQUEST_DELAY = 0.1  # 100ms between requests


class BinanceRESTClient:
    """
    Async REST client for Binance historical data backfill.

    Used during cold boot to populate the Ring Buffer with enough
    historical data for feature computation (600 bars minimum).
    Also fetches supplementary data (funding rates, open interest)
    for derivatives features.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """
        Initialize REST client.

        Args:
            api_key: Binance API key (optional for public endpoints).
            api_secret: Binance API secret (optional for public endpoints).
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self._requests_made: int = 0
        self._total_latency_ms: float = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Create or return existing HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self._api_key:
                headers['X-MBX-APIKEY'] = self._api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _get(self, url: str, params: dict = None) -> dict | list:
        """
        Perform a GET request with retry logic.

        Retries up to 3 times with exponential backoff on failure.
        """
        session = await self._ensure_session()
        last_error = None

        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                async with session.get(url, params=params) as resp:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._requests_made += 1
                    self._total_latency_ms += elapsed_ms

                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        # Rate limited — wait and retry
                        retry_after = int(resp.headers.get('Retry-After', 5))
                        logger.warning(
                            "binance_rest_rate_limited",
                            retry_after=retry_after,
                        )
                        await asyncio.sleep(retry_after)
                    else:
                        text = await resp.text()
                        logger.warning(
                            "binance_rest_error",
                            status=resp.status,
                            response=text[:500],
                            url=url,
                        )
                        last_error = f"HTTP {resp.status}: {text[:200]}"

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)
                logger.warning(
                    "binance_rest_request_failed",
                    attempt=attempt + 1,
                    error=str(e),
                )

            if attempt < 2:
                await asyncio.sleep(REQUEST_DELAY * (2 ** attempt))

        raise ConnectionError(f"Failed after 3 attempts: {last_error}")

    # =========================================================================
    # OHLCV Candle Backfill
    # =========================================================================

    async def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        limit: int = 600,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch historical kline/candlestick data.

        Binance returns max 1000 candles per request. For >1000 candles,
        multiple requests are made with pagination.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            interval: Candle interval ("1m", "5m", etc.).
            limit: Number of candles to fetch (max 1000 per request).
            start_time: Start timestamp in milliseconds.
            end_time: End timestamp in milliseconds.

        Returns:
            List of bar dictionaries, ordered oldest to newest.
        """
        url = f"{BINANCE_SPOT_BASE}/api/v3/klines"
        all_candles = []
        remaining = limit

        logger.info(
            "binance_rest_fetch_klines_start",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        while remaining > 0:
            batch_size = min(remaining, 1000)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': batch_size,
            }
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            raw = await self._get(url, params)

            if not raw:
                break

            candles = [self._parse_kline(k, symbol) for k in raw]
            all_candles.extend(candles)
            remaining -= len(candles)

            if len(raw) < batch_size:
                break  # No more data available

            # Move start_time to after the last candle for next batch
            start_time = raw[-1][0] + 1
            await asyncio.sleep(REQUEST_DELAY)

        logger.info(
            "binance_rest_fetch_klines_done",
            symbol=symbol,
            candles_fetched=len(all_candles),
        )

        return all_candles

    def _parse_kline(self, raw: list, symbol: str) -> dict:
        """
        Parse a raw Binance kline array into a bar dictionary.

        Binance kline format:
          [0]  Open time (ms)
          [1]  Open price
          [2]  High price
          [3]  Low price
          [4]  Close price
          [5]  Volume
          [6]  Close time (ms)
          [7]  Quote asset volume
          [8]  Number of trades
          [9]  Taker buy base asset volume
          [10] Taker buy quote asset volume
          [11] Ignore
        """
        bar = {
            'timestamp':   int(raw[0]),
            'open':        float(raw[1]),
            'high':        float(raw[2]),
            'low':         float(raw[3]),
            'close':       float(raw[4]),
            'volume':      float(raw[5]),
            'taker_buy':   float(raw[9]),
            'trade_count': float(raw[8]),
        }

        # Map cross-asset fields
        if symbol == "ETHUSDT":
            return {
                'timestamp':  bar['timestamp'],
                'eth_close':  bar['close'],
                'eth_volume': bar['volume'],
            }
        elif symbol == "SOLUSDT":
            return {
                'timestamp':  bar['timestamp'],
                'sol_close':  bar['close'],
                'sol_volume': bar['volume'],
            }

        return bar

    # =========================================================================
    # Derivatives Data Backfill
    # =========================================================================

    async def fetch_funding_rates(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch historical funding rates from Binance Futures.

        Funding rates are updated every 8 hours. We fetch the most
        recent rates for feature backfill.

        Args:
            symbol: Futures trading pair.
            limit: Number of records to fetch.

        Returns:
            List of funding rate records.
        """
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000),
        }

        raw = await self._get(url, params)

        return [
            {
                'timestamp': int(r['fundingTime']),
                'funding':   float(r['fundingRate']),
                'symbol':    r['symbol'],
            }
            for r in raw
        ]

    async def fetch_open_interest(
        self,
        symbol: str = "BTCUSDT",
    ) -> dict:
        """
        Fetch current open interest from Binance Futures.

        Returns a single snapshot (not historical).

        Args:
            symbol: Futures trading pair.

        Returns:
            OI record dict.
        """
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        params = {'symbol': symbol}

        raw = await self._get(url, params)

        return {
            'timestamp': int(raw.get('time', time.time() * 1000)),
            'oi':        float(raw.get('openInterest', 0)),
            'symbol':    raw.get('symbol', symbol),
        }

    async def fetch_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch historical open interest from Binance Futures.

        Args:
            symbol: Futures trading pair.
            period: Time interval ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d").
            limit: Number of records.

        Returns:
            List of OI history records.
        """
        url = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
        params = {
            'symbol': symbol,
            'period': period,
            'limit': min(limit, 500),
        }

        raw = await self._get(url, params)

        return [
            {
                'timestamp': int(r.get('timestamp', 0)),
                'oi':        float(r.get('sumOpenInterest', 0)),
            }
            for r in raw
        ]

    # =========================================================================
    # Full Backfill Orchestration
    # =========================================================================

    async def full_backfill(self, n_candles: int = 600) -> dict:
        """
        Perform a complete data backfill for cold boot.

        Fetches:
          - BTC OHLCV candles (n_candles bars)
          - ETH OHLCV candles (n_candles bars)
          - SOL OHLCV candles (n_candles bars)
          - Funding rates (last 100)
          - Open interest history (last 200)

        All fetches run concurrently for speed.

        Args:
            n_candles: Number of 1-minute candles to backfill.

        Returns:
            Dictionary with all fetched data, keyed by type.
        """
        t0 = time.perf_counter()

        logger.info("backfill_starting", n_candles=n_candles)

        # Run all fetches concurrently
        results = await asyncio.gather(
            self.fetch_klines("BTCUSDT", "1m", n_candles),
            self.fetch_klines("ETHUSDT", "1m", n_candles),
            self.fetch_klines("SOLUSDT", "1m", n_candles),
            self.fetch_funding_rates("BTCUSDT", 100),
            self.fetch_open_interest_history("BTCUSDT", "5m", 200),
            return_exceptions=True,
        )

        # Unpack results, handling any individual failures gracefully
        btc_candles = results[0] if not isinstance(results[0], Exception) else []
        eth_candles = results[1] if not isinstance(results[1], Exception) else []
        sol_candles = results[2] if not isinstance(results[2], Exception) else []
        funding = results[3] if not isinstance(results[3], Exception) else []
        oi_history = results[4] if not isinstance(results[4], Exception) else []

        # Log any failures
        for i, name in enumerate(['btc', 'eth', 'sol', 'funding', 'oi']):
            if isinstance(results[i], Exception):
                logger.error(
                    "backfill_partial_failure",
                    data_type=name,
                    error=str(results[i]),
                )

        elapsed = time.perf_counter() - t0

        logger.info(
            "backfill_completed",
            elapsed_sec=round(elapsed, 2),
            btc_candles=len(btc_candles),
            eth_candles=len(eth_candles),
            sol_candles=len(sol_candles),
            funding_records=len(funding),
            oi_records=len(oi_history),
            requests_made=self._requests_made,
        )

        return {
            'btc_candles': btc_candles,
            'eth_candles': eth_candles,
            'sol_candles': sol_candles,
            'funding': funding,
            'oi_history': oi_history,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def stats(self) -> dict:
        """Return client statistics for monitoring."""
        return {
            'requests_made': self._requests_made,
            'avg_latency_ms': round(
                self._total_latency_ms / max(self._requests_made, 1), 2
            ),
        }
