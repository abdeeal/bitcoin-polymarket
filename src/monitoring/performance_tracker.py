"""
Performance Tracker — Rolling metrics for bot health monitoring.

Tracks:
  - Rolling win rate (7-day, 30-day)
  - Feature computation latency (P50, P95, P99)
  - API call success/error rates
  - Entry rate (bets / total bars)
  - PnL tracking (daily, cumulative)

Feeds data to AlertManager when thresholds are breached.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

LATENCY_WINDOW = 500        # last N latency samples
WIN_RATE_WINDOW = 200       # last N bet outcomes
API_ERROR_WINDOW = 3600     # 1 hour of API call timestamps


@dataclass
class LatencyStats:
    """Latency percentile snapshot."""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    samples: int = 0


@dataclass
class WinRateStats:
    """Rolling win rate snapshot."""
    total: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    recent_streak: int = 0
    streak_type: str = "none"  # "win" | "loss" | "none"


@dataclass
class APIStats:
    """API call success/error tracking."""
    total_calls: int = 0
    errors: int = 0
    error_rate_pct: float = 0.0
    last_error_at: Optional[float] = None


class PerformanceTracker:
    """
    Tracks rolling performance metrics for the bot.

    Thread-safe via single-writer design (called from main event loop).

    Usage:
        tracker = PerformanceTracker()
        tracker.record_latency(42.5)
        tracker.record_outcome(won=True)
        tracker.record_api_call(success=False)

        stats = tracker.snapshot()
    """

    def __init__(self):
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._outcomes: deque[bool] = deque(maxlen=WIN_RATE_WINDOW)
        self._api_calls: deque[tuple[float, bool]] = deque(maxlen=5000)  # (timestamp, success)
        self._daily_pnl: dict[str, float] = {}  # "YYYY-MM-DD" → pnl sum
        self._cumulative_pnl: float = 0.0
        self._total_bars: int = 0
        self._total_bets: int = 0

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_latency(self, latency_ms: float) -> None:
        """Record a single feature computation latency (ms)."""
        self._latencies.append(latency_ms)

    def record_outcome(self, won: bool, pnl: float = 0.0) -> None:
        """Record a bet outcome."""
        self._outcomes.append(won)
        self._total_bets += 1
        self._cumulative_pnl += pnl

        # Daily PnL
        day = time.strftime("%Y-%m-%d")
        self._daily_pnl[day] = self._daily_pnl.get(day, 0.0) + pnl

    def record_api_call(self, success: bool) -> None:
        """Record an API call result."""
        self._api_calls.append((time.time(), success))

    def record_bar(self) -> None:
        """Record that a bar/candle was processed."""
        self._total_bars += 1

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def latency_stats(self) -> LatencyStats:
        """Get latency percentile statistics."""
        if not self._latencies:
            return LatencyStats()

        arr = np.array(self._latencies)
        return LatencyStats(
            p50_ms=round(float(np.percentile(arr, 50)), 1),
            p95_ms=round(float(np.percentile(arr, 95)), 1),
            p99_ms=round(float(np.percentile(arr, 99)), 1),
            mean_ms=round(float(arr.mean()), 1),
            samples=len(arr),
        )

    def win_rate_stats(self) -> WinRateStats:
        """Get rolling win rate and streak info."""
        if not self._outcomes:
            return WinRateStats()

        outcomes = list(self._outcomes)
        wins = sum(outcomes)
        total = len(outcomes)

        # Current streak
        streak = 0
        if outcomes:
            last_val = outcomes[-1]
            for o in reversed(outcomes):
                if o == last_val:
                    streak += 1
                else:
                    break

        return WinRateStats(
            total=total,
            wins=wins,
            losses=total - wins,
            win_rate=round(wins / max(total, 1), 4),
            recent_streak=streak,
            streak_type="win" if outcomes[-1] else "loss" if outcomes else "none",
        )

    def api_stats(self) -> APIStats:
        """Get API success/error rate for the last hour."""
        now = time.time()
        cutoff = now - API_ERROR_WINDOW

        recent = [(ts, ok) for ts, ok in self._api_calls if ts > cutoff]
        if not recent:
            return APIStats()

        total = len(recent)
        errors = sum(1 for _, ok in recent if not ok)
        last_err = None
        for ts, ok in reversed(recent):
            if not ok:
                last_err = ts
                break

        return APIStats(
            total_calls=total,
            errors=errors,
            error_rate_pct=round(errors / max(total, 1) * 100, 1),
            last_error_at=last_err,
        )

    def entry_rate(self) -> float:
        """Percentage of bars where a bet was placed."""
        if self._total_bars == 0:
            return 0.0
        return round(self._total_bets / self._total_bars * 100, 1)

    def snapshot(self) -> dict:
        """Full performance snapshot for logging/dashboard."""
        lat = self.latency_stats()
        wr = self.win_rate_stats()
        api = self.api_stats()

        return {
            'latency': {
                'p50_ms': lat.p50_ms, 'p95_ms': lat.p95_ms,
                'p99_ms': lat.p99_ms, 'mean_ms': lat.mean_ms,
                'samples': lat.samples,
            },
            'win_rate': {
                'total': wr.total, 'wins': wr.wins, 'losses': wr.losses,
                'win_rate': wr.win_rate, 'streak': wr.recent_streak,
                'streak_type': wr.streak_type,
            },
            'api': {
                'total_calls': api.total_calls, 'errors': api.errors,
                'error_rate_pct': api.error_rate_pct,
            },
            'entry_rate_pct': self.entry_rate(),
            'cumulative_pnl': round(self._cumulative_pnl, 2),
            'total_bars': self._total_bars,
            'total_bets': self._total_bets,
        }

    # ── Threshold Checks (for AlertManager) ───────────────────────────────────

    def check_latency_alert(self, budget_ms: float = 100.0) -> Optional[float]:
        """Returns P95 if exceeds budget, else None."""
        stats = self.latency_stats()
        return stats.p95_ms if stats.p95_ms > budget_ms and stats.samples >= 10 else None

    def check_win_rate_alert(self, threshold: float = 0.50) -> Optional[float]:
        """Returns win rate if below threshold (over last N bets), else None."""
        stats = self.win_rate_stats()
        if stats.total < 20:
            return None
        return stats.win_rate if stats.win_rate < threshold else None

    def check_api_error_alert(self, threshold_pct: float = 5.0) -> Optional[float]:
        """Returns error rate if exceeds threshold, else None."""
        stats = self.api_stats()
        if stats.total_calls < 10:
            return None
        return stats.error_rate_pct if stats.error_rate_pct > threshold_pct else None
