"""
Portfolio Tracker — Lifetime trade history and performance metrics.

Persists every resolved bet and computes cumulative statistics
needed for:
  - Bracket win-rate tracking
  - Drawdown monitoring
  - Reporting and dashboards
  - Model performance attribution

All methods are synchronous and append-only (no mutations to history).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BetRecord:
    """Immutable record of a single resolved bet."""
    bet_id: int
    timestamp: float
    signal: str          # Signal.name
    direction: str       # 'UP' | 'DOWN'
    is_high_conf: bool
    prob: float          # model probability
    odds: float          # decimal odds
    ev: float            # expected value edge
    bracket: int         # bracket index at time of bet
    bet_amount: float
    payout: float
    won: bool
    balance_before: float
    balance_after: float
    pnl: float           # bet_amount * (payout/bet_amount - 1) if won else -bet_amount

    def to_dict(self) -> dict:
        return {
            'bet_id': self.bet_id,
            'timestamp': self.timestamp,
            'signal': self.signal,
            'direction': self.direction,
            'is_high_conf': self.is_high_conf,
            'prob': round(self.prob, 4),
            'odds': round(self.odds, 3),
            'ev': round(self.ev, 4),
            'bracket': self.bracket,
            'bet_amount': round(self.bet_amount, 2),
            'payout': round(self.payout, 2),
            'won': self.won,
            'balance_before': round(self.balance_before, 2),
            'balance_after': round(self.balance_after, 2),
            'pnl': round(self.pnl, 2),
        }


class PortfolioTracker:
    """
    Full portfolio performance tracker.

    Records every bet and computes:
      - Win rate (overall, by tier, by bracket)
      - P&L (total, by tier)
      - Drawdown (current and max)
      - Sharpe ratio (simplified, treat each bet as a period)
      - ROI and expected value accuracy
    """

    def __init__(self, initial_balance: float = 5.00):
        self._initial_balance = initial_balance
        self._bets: list[BetRecord] = []
        self._next_id: int = 1

    def record_bet(
        self,
        signal_name: str,
        direction: str,
        is_high_conf: bool,
        prob: float,
        odds: float,
        ev: float,
        bracket: int,
        bet_amount: float,
        payout: float,
        won: bool,
        balance_before: float,
        balance_after: float,
    ) -> BetRecord:
        """Record a resolved bet and return the BetRecord."""
        pnl = payout - bet_amount if won else -bet_amount

        record = BetRecord(
            bet_id=self._next_id,
            timestamp=time.time(),
            signal=signal_name,
            direction=direction,
            is_high_conf=is_high_conf,
            prob=prob,
            odds=odds,
            ev=ev,
            bracket=bracket,
            bet_amount=bet_amount,
            payout=payout,
            won=won,
            balance_before=balance_before,
            balance_after=balance_after,
            pnl=pnl,
        )
        self._bets.append(record)
        self._next_id += 1

        logger.debug("bet_recorded", bet_id=record.bet_id, won=won, pnl=round(pnl, 2))
        return record

    # ─── Aggregate Metrics ────────────────────────────────────────────────────

    @property
    def total_bets(self) -> int:
        return len(self._bets)

    @property
    def total_wins(self) -> int:
        return sum(1 for b in self._bets if b.won)

    @property
    def win_rate(self) -> float:
        if not self._bets:
            return 0.0
        return self.total_wins / self.total_bets

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self._bets)

    @property
    def total_wagered(self) -> float:
        return sum(b.bet_amount for b in self._bets)

    @property
    def roi(self) -> float:
        if self.total_wagered == 0:
            return 0.0
        return self.total_pnl / self.total_wagered

    # ─── Tier Breakdown ───────────────────────────────────────────────────────

    def high_conf_stats(self) -> dict:
        hc = [b for b in self._bets if b.is_high_conf]
        if not hc:
            return {'count': 0, 'win_rate': 0.0, 'pnl': 0.0}
        wins = sum(1 for b in hc if b.won)
        return {
            'count': len(hc),
            'win_rate': round(wins / len(hc), 3),
            'pnl': round(sum(b.pnl for b in hc), 2),
            'avg_ev': round(np.mean([b.ev for b in hc]), 4),
        }

    def normal_stats(self) -> dict:
        norm = [b for b in self._bets if not b.is_high_conf]
        if not norm:
            return {'count': 0, 'win_rate': 0.0, 'pnl': 0.0}
        wins = sum(1 for b in norm if b.won)
        return {
            'count': len(norm),
            'win_rate': round(wins / len(norm), 3),
            'pnl': round(sum(b.pnl for b in norm), 2),
            'avg_ev': round(np.mean([b.ev for b in norm]), 4),
        }

    def bracket_stats(self) -> dict[int, dict]:
        """Per-bracket performance breakdown."""
        result = {}
        for bracket_idx in range(6):
            bets = [b for b in self._bets if b.bracket == bracket_idx]
            if not bets:
                continue
            wins = sum(1 for b in bets if b.won)
            result[bracket_idx] = {
                'count': len(bets),
                'win_rate': round(wins / len(bets), 3),
                'pnl': round(sum(b.pnl for b in bets), 2),
                'total_wagered': round(sum(b.bet_amount for b in bets), 2),
            }
        return result

    # ─── Drawdown ─────────────────────────────────────────────────────────────

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown in dollars."""
        if not self._bets:
            return 0.0
        balances = [self._initial_balance] + [b.balance_after for b in self._bets]
        peak = balances[0]
        max_dd = 0.0
        for bal in balances:
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    def current_drawdown(self) -> float:
        """Current drawdown from the most recent peak."""
        if not self._bets:
            return 0.0
        balances = [self._initial_balance] + [b.balance_after for b in self._bets]
        peak = max(balances)
        current = balances[-1]
        return round(max(0.0, peak - current), 2)

    # ─── Sharpe (simplified) ─────────────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """
        Simplified per-bet Sharpe ratio (treat each bet as one period).
        Uses PnL per bet relative to bet_amount as 'return'.
        """
        if len(self._bets) < 5:
            return 0.0
        returns = np.array([b.pnl / b.bet_amount for b in self._bets if b.bet_amount > 0])
        if returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(len(returns)))

    # ─── Consecutive loss streak ─────────────────────────────────────────────

    def current_loss_streak(self) -> int:
        """Count consecutive losses at the end of history."""
        streak = 0
        for bet in reversed(self._bets):
            if not bet.won:
                streak += 1
            else:
                break
        return streak

    # ─── Full summary ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        current_balance = self._bets[-1].balance_after if self._bets else self._initial_balance
        return {
            'total_bets': self.total_bets,
            'total_wins': self.total_wins,
            'win_rate': round(self.win_rate, 3),
            'total_pnl': round(self.total_pnl, 2),
            'total_wagered': round(self.total_wagered, 2),
            'roi': round(self.roi, 4),
            'current_balance': round(current_balance, 2),
            'max_drawdown': self.max_drawdown(),
            'current_drawdown': self.current_drawdown(),
            'sharpe_ratio': round(self.sharpe_ratio(), 3),
            'current_loss_streak': self.current_loss_streak(),
            'high_conf': self.high_conf_stats(),
            'normal': self.normal_stats(),
            'by_bracket': self.bracket_stats(),
        }
