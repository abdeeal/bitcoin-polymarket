"""
Odds Checker — Polymarket odds fetcher and EV calculator.

PRD 7.2 Gate 5: Odds EV check.
  EV = model_prob - implied_prob > 0.05

Polymarket uses binary markets with decimal odds.
This module fetches the current YES/NO prices and computes:
  - Decimal odds for UP/DOWN positions
  - Implied probability (1 / decimal_odds)
  - Expected Value edge against the model's calibrated probability

Fallback behaviour:
  - If Polymarket API is unavailable → pass gate (don't block on data gap)
  - If odds stale (>5 min) → log warning, pass gate
  - All HTTP calls are async (non-blocking to the I/O event loop)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Polymarket CLOB API endpoint
POLYMARKET_BASE_URL = "https://clob.polymarket.com"
POLYMARKET_TIMEOUT_SEC = 5.0
ODDS_STALE_AFTER_SEC = 300.0   # 5 minutes


@dataclass
class OddsData:
    """Current Polymarket odds snapshot for the BTC 5-min market."""
    market_id: str = ''

    # Raw prices (YES price range 0-1, same as probability)
    yes_price: float = 0.0    # P(BTC goes UP in 5 min) on Polymarket
    no_price: float = 0.0     # P(BTC goes DOWN)

    # Decimal odds
    odds_up: float = 0.0      # decimal odds for UP bet = 1 / yes_price
    odds_down: float = 0.0    # decimal odds for DOWN bet = 1 / no_price

    # Implied probabilities
    implied_prob_up: float = 0.0
    implied_prob_down: float = 0.0

    # Metadata
    fetched_at: float = 0.0
    source: str = 'live'      # 'live' | 'stale' | 'unavailable'

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < ODDS_STALE_AFTER_SEC

    @property
    def is_available(self) -> bool:
        return self.yes_price > 0 and self.no_price > 0

    def to_dict(self) -> dict:
        return {
            'market_id': self.market_id,
            'yes_price': round(self.yes_price, 4),
            'no_price': round(self.no_price, 4),
            'odds_up': round(self.odds_up, 4),
            'odds_down': round(self.odds_down, 4),
            'implied_prob_up': round(self.implied_prob_up, 4),
            'implied_prob_down': round(self.implied_prob_down, 4),
            'is_fresh': self.is_fresh,
            'source': self.source,
        }


class OddsChecker:
    """
    Async Polymarket odds fetcher with EV computation.

    Design:
      - fetch_odds() is called every 5-minute bar (same cadence as signals)
      - Results cached in-memory; stale odds pass the gate (fail-open)
      - EV computed against calibrated model probability per direction
      - Does NOT block the I/O event loop (uses aiohttp)
    """

    def __init__(
        self,
        market_id: str = '',
        base_url: str = POLYMARKET_BASE_URL,
        timeout: float = POLYMARKET_TIMEOUT_SEC,
        ev_minimum: float = 0.05,
    ):
        self._market_id = market_id
        self._base_url = base_url
        self._timeout = timeout
        self._ev_minimum = ev_minimum
        self._latest_odds: Optional[OddsData] = None
        self._fetch_count: int = 0
        self._fail_count: int = 0

    async def fetch_odds(self, market_id: Optional[str] = None) -> OddsData:
        """
        Fetch current YES/NO prices from Polymarket CLOB API.

        Fail-open: if API unavailable or timeout, returns unavailable OddsData
        so Gate 5 passes (we don't block bets on API data gaps).

        Returns:
            OddsData with current prices and computed odds.
        """
        mid = market_id or self._market_id
        self._fetch_count += 1

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"{self._base_url}/markets/{mid}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                    if resp.status != 200:
                        raise ValueError(f"HTTP {resp.status}")
                    data = await resp.json()
                    odds = self._parse_response(data, mid)
                    self._latest_odds = odds
                    logger.debug(
                        "odds_fetched",
                        market_id=mid,
                        yes_price=round(odds.yes_price, 4),
                        no_price=round(odds.no_price, 4),
                    )
                    return odds

        except Exception as e:
            self._fail_count += 1
            logger.warning(
                "odds_fetch_failed",
                error=str(e),
                market_id=mid,
                fallback="pass_gate",
            )
            # Return stale data if available, else unavailable
            if self._latest_odds is not None:
                stale = OddsData(**vars(self._latest_odds))
                stale.source = 'stale'
                return stale

            return OddsData(
                market_id=mid,
                source='unavailable',
                fetched_at=time.time(),
            )

    def _parse_response(self, data: dict, market_id: str) -> OddsData:
        """Parse Polymarket CLOB API market response."""
        # Polymarket CLOB returns tokens with prices
        tokens = data.get('tokens', [])
        yes_price = no_price = 0.0

        for token in tokens:
            outcome = token.get('outcome', '').upper()
            price = float(token.get('price', 0))
            if outcome == 'YES':
                yes_price = price
            elif outcome == 'NO':
                no_price = price

        # Fallback: use midpoint from orderbook if direct prices not available
        if yes_price == 0 and 'yes_price' in data:
            yes_price = float(data.get('yes_price', 0))
            no_price = float(data.get('no_price', 0))

        odds_up = 1.0 / yes_price if yes_price > 0 else 0.0
        odds_down = 1.0 / no_price if no_price > 0 else 0.0

        return OddsData(
            market_id=market_id,
            yes_price=yes_price,
            no_price=no_price,
            odds_up=odds_up,
            odds_down=odds_down,
            implied_prob_up=yes_price,
            implied_prob_down=no_price,
            fetched_at=time.time(),
            source='live',
        )

    def compute_ev(
        self,
        model_prob: float,
        direction: str,
        odds_data: Optional[OddsData] = None,
    ) -> tuple[float, bool]:
        """
        Compute Expected Value edge and whether it meets the minimum.

        EV = model_prob - implied_prob
        For UP bet: EV = P_model(UP) - P_market(UP)
        For DOWN bet: EV = P_model(DOWN) - P_market(DOWN) = (1-P_model(UP)) - P_market(DOWN)

        PRD Gate 5: EV must exceed 0.05 (5 percentage points of edge).

        Args:
            model_prob: Calibrated P(UP) from ensemble.
            direction:  'UP' or 'DOWN'.
            odds_data:  Optional OddsData; uses cached if None.

        Returns:
            (ev, passes_gate) tuple.
        """
        data = odds_data or self._latest_odds
        if data is None or not data.is_available:
            # No odds → pass gate (fail-open)
            return 0.0, True

        if direction == 'UP':
            implied = data.implied_prob_up
            ev = model_prob - implied
        else:
            implied = data.implied_prob_down
            ev = (1.0 - model_prob) - implied

        passes = ev > self._ev_minimum
        logger.debug(
            "ev_computed",
            direction=direction,
            model_prob=round(model_prob, 4),
            implied=round(implied, 4),
            ev=round(ev, 4),
            passes=passes,
        )
        return ev, passes

    def get_odds_for_signal(self, direction: str) -> float:
        """
        Return decimal odds for a given direction using cached data.

        Used by SignalResult and gate evaluation.
        """
        if self._latest_odds is None or not self._latest_odds.is_available:
            return 0.0
        if direction == 'UP':
            return self._latest_odds.odds_up
        return self._latest_odds.odds_down

    @property
    def latest_odds(self) -> Optional[OddsData]:
        return self._latest_odds

    @property
    def has_fresh_odds(self) -> bool:
        return self._latest_odds is not None and self._latest_odds.is_fresh

    def stats(self) -> dict:
        return {
            'fetch_count': self._fetch_count,
            'fail_count': self._fail_count,
            'success_rate': round(
                (self._fetch_count - self._fail_count) / max(self._fetch_count, 1), 3
            ),
            'has_fresh_odds': self.has_fresh_odds,
            'latest_odds': self._latest_odds.to_dict() if self._latest_odds else None,
        }
