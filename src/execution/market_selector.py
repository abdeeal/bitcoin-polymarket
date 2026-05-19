"""
Market Selector — Find the best active BTC prediction market for a signal.

Selects the optimal market from Polymarket's active BTC markets based on:
  - Signal direction (UP → YES, DOWN → NO)
  - Volume (prefer high-liquidity markets)
  - Odds validity (must pass EV check)
"""

from __future__ import annotations

from typing import Optional
import structlog

from src.execution.polymarket_client import PolymarketClient, Market, BetSide
from src.signal.signal_generator import Signal

logger = structlog.get_logger(__name__)

MIN_VOLUME_USD = 1000.0      # minimum 24h volume to consider
MIN_ODDS = 1.50              # minimum acceptable decimal odds
MAX_ODDS = 3.00              # maximum acceptable decimal odds


class MarketSelector:
    """
    Select the best Polymarket market for a given signal.

    Filters markets by direction, liquidity, and odds bounds.
    Returns the highest-volume valid market.
    """

    def __init__(self, client: PolymarketClient):
        self._client = client

    async def find_best_market(self, signal: Signal) -> Optional[Market]:
        """
        Find the best market for a given signal.

        Args:
            signal: The signal to find a market for.

        Returns:
            Best Market, or None if no valid market found.
        """
        try:
            markets = await self._client.get_active_btc_markets()
        except Exception as e:
            logger.error("market_fetch_failed", error=str(e))
            return None

        if not markets:
            logger.warning("no_active_markets")
            return None

        # Filter by direction
        side = self._signal_to_side(signal)
        if side is None:
            return None

        direction = side.value  # "UP" | "DOWN"
        valid = [
            m for m in markets
            if m.active
            and m.volume_24h >= MIN_VOLUME_USD
            and MIN_ODDS <= m.best_price <= MAX_ODDS
        ]

        if not valid:
            logger.warning("no_valid_markets", signal=signal.name)
            return None

        # Select highest volume
        best = max(valid, key=lambda m: m.volume_24h)
        logger.info(
            "market_selected",
            market_id=best.market_id,
            odds=best.best_price,
            volume=best.volume_24h,
        )
        return best

    @staticmethod
    def _signal_to_side(signal: Signal) -> Optional[BetSide]:
        """Map signal to BetSide."""
        if signal in (Signal.BET_UP_NORMAL, Signal.BET_UP_HIGH_CONF):
            return BetSide.UP
        if signal in (Signal.BET_DOWN_NORMAL, Signal.BET_DOWN_HIGH_CONF):
            return BetSide.DOWN
        return None
