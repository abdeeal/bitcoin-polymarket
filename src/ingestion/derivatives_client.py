"""
Derivatives Data Client — REST polling for funding, OI, IV, liquidations.

Architecture Rules:
  - Runs in the asyncio event loop (I/O thread)
  - Polls Binance Futures, Deribit, and Coinglass APIs at fixed intervals
  - Writes to Ring Buffer and persistence queue
  - NEVER performs CPU-heavy computation

Data Sources:
  - Binance Futures: Funding rate, Open Interest (every 5 minutes)
  - Deribit REST API: Implied Volatility, Put/Call ratio (every 1 minute)
  - Coinglass API: Liquidation data (every 1 minute)
"""

import asyncio
import time
from typing import Optional

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# API endpoints
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
DERIBIT_BASE = "https://www.deribit.com/api/v2"
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"

# Polling intervals
BINANCE_FUTURES_INTERVAL = 300   # 5 minutes
DERIBIT_INTERVAL = 60            # 1 minute
COINGLASS_INTERVAL = 60          # 1 minute


class DerivativesClient:
    """
    Async REST polling client for derivatives market data.

    Periodically fetches funding rates, open interest, implied volatility,
    put/call ratio, and liquidation data from multiple sources.

    All data is pushed to the Ring Buffer for feature computation
    and to the persistence queue for async DB writes.
    """

    def __init__(
        self,
        ring_buffer=None,
        persistence_queue: Optional[asyncio.Queue] = None,
        deribit_client_id: Optional[str] = None,
        deribit_client_secret: Optional[str] = None,
        coinglass_api_key: Optional[str] = None,
    ):
        """
        Initialize the derivatives data client.

        Args:
            ring_buffer: RingBuffer instance for in-memory data storage.
            persistence_queue: Async queue for fire-and-forget DB writes.
            deribit_client_id: Deribit API client ID.
            deribit_client_secret: Deribit API client secret.
            coinglass_api_key: Coinglass API key.
        """
        self._ring_buffer = ring_buffer
        self._persistence_queue = persistence_queue
        self._deribit_client_id = deribit_client_id
        self._deribit_client_secret = deribit_client_secret
        self._coinglass_api_key = coinglass_api_key

        self._session: Optional[aiohttp.ClientSession] = None
        self._running: bool = False

        # Statistics
        self._poll_counts = {
            'funding': 0,
            'oi': 0,
            'deribit': 0,
            'liquidations': 0,
        }
        self._error_counts = {
            'funding': 0,
            'oi': 0,
            'deribit': 0,
            'liquidations': 0,
        }

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Create or return existing HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get(self, url: str, params: dict = None,
                   headers: dict = None) -> Optional[dict | list]:
        """Perform GET request with error handling."""
        session = await self._ensure_session()
        try:
            async with session.get(url, params=params,
                                   headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.warning(
                        "derivatives_api_error",
                        url=url,
                        status=resp.status,
                        response=text[:300],
                    )
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("derivatives_request_failed", url=url, error=str(e))
            return None

    # =========================================================================
    # Polling Loops
    # =========================================================================

    async def start_polling(self) -> None:
        """
        Start all polling loops as concurrent tasks.

        Each data source runs in its own polling loop with independent
        intervals and error handling. Failure in one source does not
        affect others.
        """
        self._running = True
        logger.info("derivatives_polling_starting")

        await asyncio.gather(
            self._poll_funding_rate(),
            self._poll_open_interest(),
            self._poll_deribit(),
            self._poll_liquidations(),
            return_exceptions=True,
        )

    async def stop_polling(self) -> None:
        """Stop all polling loops."""
        self._running = False
        await self._cleanup()

    # =========================================================================
    # Binance Futures — Funding Rate
    # =========================================================================

    async def _poll_funding_rate(self) -> None:
        """Poll Binance Futures funding rate every 5 minutes."""
        while self._running:
            try:
                data = await self._fetch_funding_rate()
                if data:
                    self._poll_counts['funding'] += 1
                    await self._update_ring_buffer({'funding': data['funding']})
                    await self._persist('funding', data)
            except Exception as e:
                self._error_counts['funding'] += 1
                logger.error("funding_rate_poll_error", error=str(e))

            await asyncio.sleep(BINANCE_FUTURES_INTERVAL)

    async def _fetch_funding_rate(self) -> Optional[dict]:
        """Fetch current funding rate from Binance Futures."""
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
        params = {'symbol': 'BTCUSDT'}

        raw = await self._get(url, params)
        if not raw:
            return None

        return {
            'timestamp': int(raw.get('time', time.time() * 1000)),
            'funding': float(raw.get('lastFundingRate', 0)),
            'mark_price': float(raw.get('markPrice', 0)),
            'index_price': float(raw.get('indexPrice', 0)),
        }

    # =========================================================================
    # Binance Futures — Open Interest
    # =========================================================================

    async def _poll_open_interest(self) -> None:
        """Poll Binance Futures open interest every 5 minutes."""
        while self._running:
            try:
                data = await self._fetch_open_interest()
                if data:
                    self._poll_counts['oi'] += 1
                    await self._update_ring_buffer({'oi': data['oi']})
                    await self._persist('oi', data)
            except Exception as e:
                self._error_counts['oi'] += 1
                logger.error("open_interest_poll_error", error=str(e))

            await asyncio.sleep(BINANCE_FUTURES_INTERVAL)

    async def _fetch_open_interest(self) -> Optional[dict]:
        """Fetch current open interest from Binance Futures."""
        url = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
        params = {'symbol': 'BTCUSDT'}

        raw = await self._get(url, params)
        if not raw:
            return None

        return {
            'timestamp': int(time.time() * 1000),
            'oi': float(raw.get('openInterest', 0)),
        }

    # =========================================================================
    # Deribit — Implied Volatility & Put/Call Ratio
    # =========================================================================

    async def _poll_deribit(self) -> None:
        """Poll Deribit API every 1 minute for IV and put/call data."""
        while self._running:
            try:
                data = await self._fetch_deribit_data()
                if data:
                    self._poll_counts['deribit'] += 1
                    await self._update_ring_buffer({
                        'iv_25delta': data.get('iv_25delta', 0),
                        'put_call_ratio': data.get('put_call_ratio', 0),
                    })
                    await self._persist('deribit', data)
            except Exception as e:
                self._error_counts['deribit'] += 1
                logger.error("deribit_poll_error", error=str(e))

            await asyncio.sleep(DERIBIT_INTERVAL)

    async def _fetch_deribit_data(self) -> Optional[dict]:
        """
        Fetch BTC volatility index and options summary from Deribit.

        Uses the public API endpoints (no auth required for market data).
        """
        # Fetch BTC volatility index (DVOL)
        dvol_url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
        dvol_params = {
            'currency': 'BTC',
            'resolution': 60,  # 1-hour resolution
            'start_timestamp': int((time.time() - 3600) * 1000),
            'end_timestamp': int(time.time() * 1000),
        }

        dvol_raw = await self._get(dvol_url, dvol_params)

        iv_25delta = 0.0
        if dvol_raw and 'result' in dvol_raw:
            data_points = dvol_raw['result'].get('data', [])
            if data_points:
                # Last data point: [timestamp, open, high, low, close]
                iv_25delta = data_points[-1][4] if len(data_points[-1]) > 4 else 0

        # Fetch options book summary for put/call ratio
        book_url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"
        book_params = {
            'currency': 'BTC',
            'kind': 'option',
        }

        book_raw = await self._get(book_url, book_params)

        put_oi = 0.0
        call_oi = 0.0
        if book_raw and 'result' in book_raw:
            for instrument in book_raw['result']:
                name = instrument.get('instrument_name', '')
                oi = float(instrument.get('open_interest', 0))
                if '-P' in name:
                    put_oi += oi
                elif '-C' in name:
                    call_oi += oi

        put_call_ratio = put_oi / (call_oi + 1e-10)

        return {
            'timestamp': int(time.time() * 1000),
            'iv_25delta': iv_25delta,
            'put_call_ratio': put_call_ratio,
            'put_oi': put_oi,
            'call_oi': call_oi,
        }

    # =========================================================================
    # Coinglass — Liquidation Data
    # =========================================================================

    async def _poll_liquidations(self) -> None:
        """Poll Coinglass API every 1 minute for liquidation data."""
        if not self._coinglass_api_key:
            logger.info("coinglass_api_key_not_set_skipping_liquidation_polling")
            return

        while self._running:
            try:
                data = await self._fetch_liquidations()
                if data:
                    self._poll_counts['liquidations'] += 1
                    await self._update_ring_buffer({
                        'liq_long': data.get('liq_long', 0),
                        'liq_short': data.get('liq_short', 0),
                    })
                    await self._persist('liquidation', data)
            except Exception as e:
                self._error_counts['liquidations'] += 1
                logger.error("liquidation_poll_error", error=str(e))

            await asyncio.sleep(COINGLASS_INTERVAL)

    async def _fetch_liquidations(self) -> Optional[dict]:
        """Fetch BTC liquidation data from Coinglass."""
        url = f"{COINGLASS_BASE}/liquidation_info"
        params = {
            'symbol': 'BTC',
            'time_type': 1,  # 1 = last 1 hour
        }
        headers = {
            'coinglassSecret': self._coinglass_api_key,
        }

        raw = await self._get(url, params, headers)
        if not raw or raw.get('code') != '0':
            return None

        data = raw.get('data', [])
        if not data:
            return None

        # Sum across all exchanges
        total_long_liq = 0.0
        total_short_liq = 0.0
        for exchange in data:
            total_long_liq += float(exchange.get('longLiquidationUsd', 0))
            total_short_liq += float(exchange.get('shortLiquidationUsd', 0))

        return {
            'timestamp': int(time.time() * 1000),
            'liq_long': total_long_liq,
            'liq_short': total_short_liq,
            'liq_ratio': total_long_liq / (total_short_liq + 1e-10),
        }

    # =========================================================================
    # One-shot Fetch (for boot sequence)
    # =========================================================================

    async def fetch_all_once(self) -> dict:
        """
        Fetch all derivatives data once (used during boot).

        Returns all data in a single dict for Ring Buffer population.
        """
        results = await asyncio.gather(
            self._fetch_funding_rate(),
            self._fetch_open_interest(),
            self._fetch_deribit_data(),
            self._fetch_liquidations() if self._coinglass_api_key else asyncio.coroutine(lambda: None)(),
            return_exceptions=True,
        )

        combined = {}
        for result in results:
            if isinstance(result, dict):
                combined.update(result)

        return combined

    # =========================================================================
    # Ring Buffer & Persistence Integration
    # =========================================================================

    async def _update_ring_buffer(self, data: dict) -> None:
        """Update the latest bar in the Ring Buffer with derivatives data."""
        if self._ring_buffer:
            self._ring_buffer.update_latest(data)

    async def _persist(self, data_type: str, data: dict) -> None:
        """Push data to persistence queue (fire-and-forget)."""
        if self._persistence_queue:
            try:
                self._persistence_queue.put_nowait((data_type, data))
            except asyncio.QueueFull:
                pass  # Non-blocking: drop if queue is full

    async def _cleanup(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def stats(self) -> dict:
        """Return polling statistics for monitoring."""
        return {
            'poll_counts': dict(self._poll_counts),
            'error_counts': dict(self._error_counts),
            'running': self._running,
        }
