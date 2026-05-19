"""
Backtester — Full simulation engine with cooling down.

PRD 11.1: Walk bar-by-bar through feature_df, mirror production logic:
  - Features always computed (even during cooling down)
  - Cooling down skips execution only, not data
  - 2 consecutive losses → 30-min cooling down
  - Compounding engine manages bracket progression
  - Cost model deducts fees + slippage

Usage:
    bt = Backtester(initial_capital=10.0)
    result = bt.run(feature_df, model_fn)
    print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable

import numpy as np
import structlog

from src.sizing.compounding_engine import CompoundingEngine
from src.signal.signal_generator import Signal, SignalGenerator, BotState
from src.backtest.cost_model import net_payout

logger = structlog.get_logger(__name__)

# Cooling down duration in bars (30 min / 5 min per bar = 6 bars)
COOLING_DOWN_BARS = 6
CONSECUTIVE_LOSS_TRIGGER = 2


@dataclass
class BetRecord:
    """Single bet in the backtest log."""
    bar_index: int
    timestamp: float
    signal: str
    prob: float
    bet_size: float
    odds: float
    won: bool
    pnl: float
    balance: float
    bracket: int
    cooling_down: bool = False


@dataclass
class BacktestResult:
    """Complete backtest output with full diagnostics."""
    initial_capital: float = 10.0
    final_capital: float = 10.0
    total_bars: int = 0
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    cooling_down_events: int = 0
    cooling_down_bars_skipped: int = 0
    max_drawdown_pct: float = 0.0
    peak_balance: float = 10.0
    lowest_balance: float = 10.0
    bet_log: list[BetRecord] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.total_wins / max(self.total_bets, 1)

    @property
    def total_return_pct(self) -> float:
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def avg_pnl(self) -> float:
        return (self.final_capital - self.initial_capital) / max(self.total_bets, 1)

    @property
    def profit_factor(self) -> float:
        wins = sum(b.pnl for b in self.bet_log if b.pnl > 0)
        losses = abs(sum(b.pnl for b in self.bet_log if b.pnl < 0))
        return wins / max(losses, 1e-10)

    @property
    def entry_rate(self) -> float:
        """Percentage of bars where a bet was placed."""
        return self.total_bets / max(self.total_bars, 1) * 100

    def summary(self) -> dict:
        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(self.final_capital, 2),
            'total_return_pct': round(self.total_return_pct, 1),
            'total_bars': self.total_bars,
            'total_bets': self.total_bets,
            'win_rate': round(self.win_rate, 4),
            'entry_rate_pct': round(self.entry_rate, 1),
            'avg_pnl': round(self.avg_pnl, 4),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 1),
            'peak_balance': round(self.peak_balance, 2),
            'lowest_balance': round(self.lowest_balance, 2),
            'cooling_down_events': self.cooling_down_events,
            'cooling_down_bars_skipped': self.cooling_down_bars_skipped,
            'elapsed_sec': round(self.elapsed_sec, 2),
        }


class Backtester:
    """
    Bar-by-bar backtesting engine that mirrors production logic.

    Key behaviors:
      - Features always computed (state-agnostic, matches production)
      - Cooling down: 2 consecutive losses → skip next 6 bars (30 min)
      - Compounding: bracket progression based on balance
      - Cost model: Polymarket 2% fee + 5bps slippage

    Args:
        initial_capital: Starting balance.
        odds: Fixed decimal odds for all bets (e.g., 1.90).
        signal_threshold: Minimum probability to generate a signal.
        high_conf_threshold: Probability for HIGH_CONF tier.
    """

    def __init__(
        self,
        initial_capital: float = 10.0,
        odds: float = 1.90,
        signal_threshold: float = 0.55,
        high_conf_threshold: float = 0.70,
    ):
        self.initial_capital = initial_capital
        self.odds = odds
        self.signal_threshold = signal_threshold
        self.high_conf_threshold = high_conf_threshold

    def run(
        self,
        probabilities: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest over an array of model probabilities.

        Args:
            probabilities: Array of P(up) from model, shape (n_bars,).
            timestamps:    Optional Unix timestamps for each bar.
            actuals:       Optional actual outcomes (1=up, 0=down).
                           If None, outcomes are simulated from probabilities.

        Returns:
            BacktestResult with full bet log and diagnostics.
        """
        t0 = time.perf_counter()
        n = len(probabilities)

        if timestamps is None:
            timestamps = np.arange(n, dtype=float) * 300  # 5-min intervals

        ce = CompoundingEngine(self.initial_capital)
        gen = SignalGenerator()

        consecutive_losses = 0
        cooling_down_until = -1
        cooling_down_events = 0
        cooling_down_bars = 0
        peak = self.initial_capital
        lowest = self.initial_capital

        result = BacktestResult(
            initial_capital=self.initial_capital,
            total_bars=n,
        )

        for i in range(n):
            prob = float(probabilities[i])
            ts = float(timestamps[i])

            # ── Cooling down check ──
            if i < cooling_down_until:
                cooling_down_bars += 1
                continue

            # ── Signal generation ──
            sig_result = gen.generate_signal(
                prob=prob,
                filters_ok=True,
                bot_state=BotState.ACTIVE,
            )

            if sig_result.signal == Signal.NO_BET:
                continue

            # ── Bet sizing ──
            bet_size = ce.resolve_bet_size(
                sig_result.signal,
                prob=prob,
                odds=self.odds
            )
            if bet_size <= 0:
                continue

            # ── Outcome ──
            if actuals is not None:
                won = bool(actuals[i] == 1) if sig_result.signal in (
                    Signal.BET_UP_NORMAL, Signal.BET_UP_HIGH_CONF
                ) else bool(actuals[i] == 0)
            else:
                # Simulate: P(win) = prob for UP bets, (1-prob) for DOWN
                won = np.random.rand() < prob

            # ── PnL ──
            pnl = net_payout(bet_size, self.odds, won)
            ce.on_bet_resolved(
                won=won,
                bet_amount=bet_size,
                payout=bet_size + pnl if won else 0.0,
            )

            # Track balance
            balance = ce.balance
            peak = max(peak, balance)
            lowest = min(lowest, balance)

            # Drawdown
            if peak > 0:
                dd = (peak - balance) / peak * 100
                result.max_drawdown_pct = max(result.max_drawdown_pct, dd)

            # ── Record ──
            result.bet_log.append(BetRecord(
                bar_index=i,
                timestamp=ts,
                signal=sig_result.signal.name,
                prob=prob,
                bet_size=bet_size,
                odds=self.odds,
                won=won,
                pnl=pnl,
                balance=balance,
                bracket=ce.current_bracket.index,
            ))

            if won:
                result.total_wins += 1
                consecutive_losses = 0
            else:
                result.total_losses += 1
                consecutive_losses += 1
                if consecutive_losses >= CONSECUTIVE_LOSS_TRIGGER:
                    cooling_down_until = i + COOLING_DOWN_BARS
                    cooling_down_events += 1
                    consecutive_losses = 0

        result.total_bets = len(result.bet_log)
        result.final_capital = ce.balance
        result.peak_balance = peak
        result.lowest_balance = lowest
        result.cooling_down_events = cooling_down_events
        result.cooling_down_bars_skipped = cooling_down_bars
        result.elapsed_sec = time.perf_counter() - t0

        logger.info(
            "backtest_completed",
            bets=result.total_bets,
            win_rate=round(result.win_rate, 3),
            final_capital=round(result.final_capital, 2),
            cooling_downs=cooling_down_events,
        )

        return result
