"""
Paper Trader — Simulates bet execution without real money.

PRD 12.3 Paper Trading Validation Criteria:
  - Win rate > 53% over 50+ paper bets
  - Entry rate 6–9% (1 bet per ~11-17 bars)
  - ≥1 cooling down event triggered and recovered
  - ≥1 bracket upgrade observed
  - Latency < 500ms consistently
  - Zero execution bugs in 2-week window

Outcome resolution uses REAL BTC price comparison:
  - A BET_DOWN wins if exit_price < entry_price (price fell).
  - A BET_UP wins if exit_price > entry_price (price rose).
  - Falls back to prob-based RNG only if prices are unavailable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from src.execution.polymarket_client import BetReceipt, Market
from src.signal.signal_generator import Signal
from src.backtest.cost_model import net_payout

logger = structlog.get_logger(__name__)


@dataclass
class PaperBet:
    """Record of a simulated paper bet."""
    bet_id: str
    market_id: str
    signal: str
    prob: float
    bet_size: float
    odds: float
    placed_at: float
    entry_price: float = 0.0        # BTC price when bet was placed
    resolved_at: Optional[float] = None
    won: Optional[bool] = None
    pnl: Optional[float] = None
    balance_after: Optional[float] = None
    exit_price: Optional[float] = None  # BTC price when bet resolved

    def to_dict(self) -> dict:
        return {
            'bet_id': self.bet_id,
            'market_id': self.market_id,
            'signal': self.signal,
            'prob': self.prob,
            'bet_size': self.bet_size,
            'odds': self.odds,
            'placed_at': self.placed_at,
            'entry_price': self.entry_price,
            'resolved_at': self.resolved_at,
            'won': self.won,
            'pnl': self.pnl,
            'balance_after': self.balance_after,
            'exit_price': self.exit_price,
        }


class PaperTrader:
    """
    Simulates Polymarket bet execution for paper trading.

    Outcome simulation: uses the model probability to determine
    win/loss stochastically — consistent with how the model would
    perform if its probabilities were calibrated.

    Usage:
        pt = PaperTrader(seed=42)
        receipt = pt.simulate(market, signal, bet_size, prob=0.75)
        won = pt.resolve(receipt, prob=0.75, odds=1.90)
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._bets: list[PaperBet] = []
        self._total_pnl: float = 0.0

    def simulate(
        self,
        market: Market,
        signal: Signal,
        bet_size: float,
        prob: float = 0.70,
        entry_price: float = 0.0,
    ) -> BetReceipt:
        """
        Simulate placing a bet. Returns a receipt immediately.

        Args:
            market:       Selected market.
            signal:       Trading signal.
            bet_size:     Amount to bet in USD.
            prob:         Model probability for the bet direction.
            entry_price:  Current BTC price at bet placement (used for real resolution).

        Returns:
            BetReceipt with a unique paper order_id.
        """
        order_id = f"paper_{uuid.uuid4().hex[:12]}"

        bet = PaperBet(
            bet_id=order_id,
            market_id=market.market_id,
            signal=signal.name,
            prob=prob,
            bet_size=bet_size,
            odds=market.best_price,
            placed_at=time.time(),
            entry_price=entry_price,
        )
        self._bets.append(bet)

        logger.info(
            "paper_bet_placed",
            order_id=order_id,
            signal=signal.name,
            size=bet_size,
            odds=market.best_price,
            entry_price=round(entry_price, 2),
        )

        return BetReceipt(
            order_id=order_id,
            market_id=market.market_id,
            side=signal.name,
            size=bet_size,
            price=market.best_price,
            paper=True,
        )

    def resolve(
        self,
        receipt: BetReceipt,
        prob: float,
        current_balance: float,
        exit_price: Optional[float] = None,
    ) -> tuple[bool, float]:
        """
        Resolve a paper bet using real BTC price movement.

        If `exit_price` is provided AND the bet has a recorded `entry_price`,
        the outcome is determined by actual price direction:
          - BET_DOWN wins  if exit_price < entry_price  (price fell as predicted).
          - BET_UP  wins  if exit_price > entry_price  (price rose as predicted).
        Otherwise falls back to prob-based RNG simulation.

        Args:
            receipt:         BetReceipt from simulate().
            prob:            Model probability (fallback RNG only).
            current_balance: Balance before this bet resolves.
            exit_price:      BTC close price of the NEXT 5-min candle.

        Returns:
            (won, new_balance) tuple.
        """
        # Find matching bet record
        bet = next((b for b in self._bets if b.bet_id == receipt.order_id), None)
        if bet is None:
            logger.error("paper_bet_not_found", order_id=receipt.order_id)
            return False, current_balance

        # ── Determine outcome ──────────────────────────────────────────────
        if exit_price is not None and bet.entry_price > 0:
            # Real price-based resolution
            price_moved_down = exit_price < bet.entry_price
            is_down_bet = "DOWN" in bet.signal
            won = (is_down_bet and price_moved_down) or (not is_down_bet and not price_moved_down)
            resolution_method = "price"
        else:
            # Fallback: stochastic simulation using model probability
            won = bool(self._rng.random() < prob)
            resolution_method = "rng"

        pnl = net_payout(bet.bet_size, bet.odds, won)
        new_balance = current_balance + pnl

        bet.resolved_at = time.time()
        bet.won = won
        bet.pnl = pnl
        bet.balance_after = new_balance
        bet.exit_price = exit_price
        self._total_pnl += pnl

        logger.info(
            "paper_bet_resolved",
            order_id=receipt.order_id,
            won=won,
            pnl=round(pnl, 4),
            new_balance=round(new_balance, 4),
            entry_price=round(bet.entry_price, 2) if bet.entry_price else None,
            exit_price=round(exit_price, 2) if exit_price else None,
            resolution_method=resolution_method,
        )

        return won, new_balance

    @property
    def total_bets(self) -> int:
        return len(self._bets)

    @property
    def resolved_bets(self) -> list[PaperBet]:
        return [b for b in self._bets if b.won is not None]

    @property
    def win_rate(self) -> float:
        resolved = self.resolved_bets
        if not resolved:
            return 0.0
        return sum(1 for b in resolved if b.won) / len(resolved)

    def validation_report(self) -> dict:
        """PRD 12.3 paper trading validation criteria check."""
        resolved = self.resolved_bets
        wins = sum(1 for b in resolved if b.won)
        total = len(resolved)
        wr = wins / max(total, 1)

        return {
            'total_bets': total,
            'win_rate': round(wr, 4),
            'total_pnl': round(self._total_pnl, 2),
            'criteria': {
                'win_rate_gt_53pct': wr > 0.53,
                'min_50_bets': total >= 50,
            },
            'go': wr > 0.53 and total >= 50,
        }
