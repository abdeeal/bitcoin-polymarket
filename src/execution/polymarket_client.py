"""
Polymarket CLOB Client — Async REST client for Polymarket order book.

PRD 12.1 Spec:
  - get_active_btc_markets() → list[Market]
  - get_market_odds(market_id) → float
  - place_bet(market_id, side, size, price) → BetReceipt
  - get_bet_status(order_id) → BetStatus
  - Retry up to 3× with exponential backoff

Circuit Breaker (PRD 12.4):
  - Track consecutive API failures
  - After 5 consecutive failures → raise CircuitOpenError
  - Circuit resets after successful call
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0   # seconds, doubles each retry
CIRCUIT_BREAKER_THRESHOLD = 5


class BetSide(Enum):
    UP   = "UP"
    DOWN = "DOWN"


class BetStatusValue(Enum):
    PENDING  = "PENDING"
    MATCHED  = "MATCHED"
    WON      = "WON"
    LOST     = "LOST"
    CANCELED = "CANCELED"


@dataclass
class Market:
    market_id: str
    question: str
    side: str          # "YES" | "NO"
    best_price: float  # decimal odds (e.g. 1.90)
    volume_24h: float
    active: bool = True


@dataclass
class BetReceipt:
    order_id: str
    market_id: str
    side: str
    size: float
    price: float
    placed_at: float = field(default_factory=time.time)
    paper: bool = False


@dataclass
class BetStatus:
    order_id: str
    status: BetStatusValue
    filled_size: float = 0.0
    settled_at: Optional[float] = None


class CircuitOpenError(Exception):
    """Raised when circuit breaker trips (5 consecutive API failures)."""
    pass


class PolymarketClient:
    """
    Async REST client for Polymarket CLOB API.

    Implements retry-with-backoff and circuit breaker.
    In paper/stub mode, all methods return simulated responses.

    Args:
        api_key:     Polymarket API key (None for paper mode).
        base_url:    API base URL.
        paper_mode:  If True, no real API calls made.
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        paper_mode: bool = True,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._paper_mode = paper_mode
        self._consecutive_failures = 0
        self._total_calls = 0
        self._total_errors = 0

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def _check_circuit(self) -> None:
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            raise CircuitOpenError(
                f"Circuit open: {self._consecutive_failures} consecutive API failures"
            )

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        self._total_calls += 1

    def _on_failure(self, error: Exception) -> None:
        self._consecutive_failures += 1
        self._total_errors += 1
        logger.warning(
            "api_call_failed",
            error=str(error),
            consecutive=self._consecutive_failures,
        )

    # ── Retry Helper ─────────────────────────────────────────────────────────

    async def _with_retry(self, coro_fn, *args, **kwargs):
        """Execute an async function with up to 3 retries + exponential backoff."""
        self._check_circuit()
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                result = await coro_fn(*args, **kwargs)
                self._on_success()
                return result
            except CircuitOpenError:
                raise
            except Exception as e:
                last_error = e
                self._on_failure(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.info("retry_scheduled", attempt=attempt + 1, delay=delay)
                    await asyncio.sleep(delay)
        raise last_error

    # ── API Methods ───────────────────────────────────────────────────────────

    async def get_active_btc_markets(self) -> list[Market]:
        """Fetch active BTC price prediction markets."""
        if self._paper_mode:
            return self._stub_btc_markets()

        async def _call():
            # Production: GET /markets?tag=BTC&active=true
            raise NotImplementedError("Set paper_mode=False with real API key")

        return await self._with_retry(_call)

    async def get_market_odds(self, market_id: str) -> float:
        """Get best available decimal odds for a market."""
        if self._paper_mode:
            return 1.90  # stub

        async def _call():
            raise NotImplementedError("Set paper_mode=False with real API key")

        return await self._with_retry(_call)

    async def place_bet(
        self,
        market_id: str,
        side: BetSide,
        size: float,
        price: float,
    ) -> BetReceipt:
        """
        Place a bet on Polymarket.

        Args:
            market_id: Market identifier.
            side:      BetSide.UP or BetSide.DOWN.
            size:      Bet amount in USD.
            price:     Decimal odds at time of placement.

        Returns:
            BetReceipt with order_id for status tracking.
        """
        if self._paper_mode:
            return BetReceipt(
                order_id=f"paper_{int(time.time()*1000)}",
                market_id=market_id,
                side=side.value,
                size=size,
                price=price,
                paper=True,
            )

        async def _call():
            raise NotImplementedError("Set paper_mode=False with real API key")

        return await self._with_retry(_call)

    async def get_bet_status(self, order_id: str) -> BetStatus:
        """Poll order status until settled."""
        if self._paper_mode:
            return BetStatus(
                order_id=order_id,
                status=BetStatusValue.MATCHED,
                filled_size=1.0,
            )

        async def _call():
            raise NotImplementedError("Set paper_mode=False with real API key")

        return await self._with_retry(_call)

    # ── Stubs ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _stub_btc_markets() -> list[Market]:
        return [
            Market(
                market_id="btc-5m-up-001",
                question="Will BTC be higher in 5 minutes?",
                side="YES",
                best_price=1.90,
                volume_24h=50_000.0,
            ),
        ]

    def stats(self) -> dict:
        return {
            'total_calls': self._total_calls,
            'total_errors': self._total_errors,
            'consecutive_failures': self._consecutive_failures,
            'circuit_open': self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD,
            'paper_mode': self._paper_mode,
        }
