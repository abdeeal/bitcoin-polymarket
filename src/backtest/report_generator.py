"""
Report Generator — Post-backtest analytics and reporting.

Produces:
  - Summary statistics (win rate, PnL, drawdown, Sharpe-like ratio)
  - Per-period breakdown (daily, weekly, monthly)
  - Cooling down analysis
  - Bracket progression timeline
  - Streak analysis (consecutive wins/losses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StreakStats:
    """Consecutive win/loss streak analysis."""
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    total_win_streaks: int = 0
    total_loss_streaks: int = 0


@dataclass
class Report:
    """Structured backtest report."""
    summary: dict = field(default_factory=dict)
    streaks: Optional[StreakStats] = None
    pnl_by_signal: dict = field(default_factory=dict)
    bracket_transitions: list = field(default_factory=list)


class ReportGenerator:
    """Generate analytics reports from BacktestResult."""

    def generate(self, result) -> Report:
        """
        Generate a full report from a BacktestResult.

        Args:
            result: BacktestResult from Backtester.run()

        Returns:
            Report with summary, streaks, signal analysis.
        """
        report = Report()
        report.summary = result.summary()

        # Streak analysis
        report.streaks = self._compute_streaks(result.bet_log)

        # PnL by signal type
        report.pnl_by_signal = self._pnl_by_signal(result.bet_log)

        # Bracket transitions
        report.bracket_transitions = self._bracket_transitions(result.bet_log)

        # Sharpe-like ratio
        if result.bet_log:
            pnls = np.array([b.pnl for b in result.bet_log])
            if pnls.std() > 0:
                report.summary['sharpe_ratio'] = round(
                    float(pnls.mean() / pnls.std() * np.sqrt(252 * 288 / max(len(pnls), 1))),
                    2,
                )
            else:
                report.summary['sharpe_ratio'] = 0.0

        return report

    def _compute_streaks(self, bet_log: list) -> StreakStats:
        """Compute consecutive win/loss streak statistics."""
        stats = StreakStats()
        if not bet_log:
            return stats

        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None

        for bet in bet_log:
            if bet.won:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_type == 'loss' and current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'win'
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_type == 'win' and current_streak > 0:
                        win_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'loss'

        # Flush last streak
        if current_type == 'win' and current_streak > 0:
            win_streaks.append(current_streak)
        elif current_type == 'loss' and current_streak > 0:
            loss_streaks.append(current_streak)

        stats.max_win_streak = max(win_streaks) if win_streaks else 0
        stats.max_loss_streak = max(loss_streaks) if loss_streaks else 0
        stats.avg_win_streak = float(np.mean(win_streaks)) if win_streaks else 0.0
        stats.avg_loss_streak = float(np.mean(loss_streaks)) if loss_streaks else 0.0
        stats.total_win_streaks = len(win_streaks)
        stats.total_loss_streaks = len(loss_streaks)

        return stats

    def _pnl_by_signal(self, bet_log: list) -> dict:
        """Aggregate PnL and win rate by signal type."""
        by_signal: dict[str, dict] = {}

        for bet in bet_log:
            sig = bet.signal
            if sig not in by_signal:
                by_signal[sig] = {'count': 0, 'wins': 0, 'total_pnl': 0.0}
            by_signal[sig]['count'] += 1
            by_signal[sig]['wins'] += int(bet.won)
            by_signal[sig]['total_pnl'] += bet.pnl

        for sig, data in by_signal.items():
            data['win_rate'] = round(data['wins'] / max(data['count'], 1), 4)
            data['avg_pnl'] = round(data['total_pnl'] / max(data['count'], 1), 4)
            data['total_pnl'] = round(data['total_pnl'], 2)

        return by_signal

    def _bracket_transitions(self, bet_log: list) -> list:
        """Record bracket changes during the backtest."""
        transitions = []
        prev_bracket = None

        for bet in bet_log:
            if prev_bracket is not None and bet.bracket != prev_bracket:
                transitions.append({
                    'bar_index': bet.bar_index,
                    'from_bracket': prev_bracket,
                    'to_bracket': bet.bracket,
                    'balance': bet.balance,
                })
            prev_bracket = bet.bracket

        return transitions
