"""
Bet Logger — Async structured logging of every bet to DB queue.

Persists bet records to TimescaleDB via the async persistence queue.
Provides in-memory recent-bets buffer for dashboard queries.

Schema target: bet_log table (see schema.sql)
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

MAX_RECENT_BETS = 200   # in-memory buffer size


@dataclass
class BetLogEntry:
    """A fully resolved bet ready to persist."""
    timestamp: float
    market_id: str
    signal: str
    confidence_tier: str
    model_prob: float
    bet_size: float
    odds: float
    outcome: Optional[bool]    # True=won, False=lost, None=pending
    pnl: Optional[float]
    balance_before: float
    balance_after: Optional[float]
    bracket_index: int
    paper_mode: bool
    feature_snapshot: dict = field(default_factory=dict)

    def to_db_row(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'market_id': self.market_id,
            'signal': self.signal,
            'confidence_tier': self.confidence_tier,
            'model_prob': self.model_prob,
            'bet_size': self.bet_size,
            'odds': self.odds,
            'outcome': self.outcome,
            'pnl': self.pnl,
            'balance_before': self.balance_before,
            'balance_after': self.balance_after,
            'bracket_index': self.bracket_index,
            'paper_mode': self.paper_mode,
            'feature_snapshot': self.feature_snapshot,
        }


class BetLogger:
    """
    Logs every bet to the async DB persistence queue.

    Maintains an in-memory deque of recent bets for fast
    dashboard queries without hitting the database.

    Usage:
        logger = BetLogger(persistence_queue=queue)
        entry = logger.log_placed(signal, prob, size, odds, balance, bracket)
        # ... after resolution:
        logger.log_resolved(entry, won=True, pnl=0.88, new_balance=10.88)
    """

    def __init__(
        self,
        persistence_queue: Optional[asyncio.Queue] = None,
        paper_mode: bool = True,
    ):
        self._queue = persistence_queue
        self._paper_mode = paper_mode
        self._recent: deque[BetLogEntry] = deque(maxlen=MAX_RECENT_BETS)
        self._total_logged = 0

    def log_placed(
        self,
        market_id: str,
        signal: str,
        confidence_tier: str,
        model_prob: float,
        bet_size: float,
        odds: float,
        balance_before: float,
        bracket_index: int,
        features: Optional[dict] = None,
    ) -> BetLogEntry:
        """Create a pending bet log entry (before resolution)."""
        entry = BetLogEntry(
            timestamp=time.time(),
            market_id=market_id,
            signal=signal,
            confidence_tier=confidence_tier,
            model_prob=model_prob,
            bet_size=bet_size,
            odds=odds,
            outcome=None,
            pnl=None,
            balance_before=balance_before,
            balance_after=None,
            bracket_index=bracket_index,
            paper_mode=self._paper_mode,
            feature_snapshot=features or {},
        )
        self._recent.append(entry)
        logger.debug("bet_placed_logged", signal=signal, size=bet_size)
        return entry

    def log_resolved(
        self,
        entry: BetLogEntry,
        won: bool,
        pnl: float,
        new_balance: float,
    ) -> None:
        """Update a placed entry with resolution result and push to DB queue."""
        entry.outcome = won
        entry.pnl = pnl
        entry.balance_after = new_balance
        self._total_logged += 1

        if self._queue is not None:
            try:
                self._queue.put_nowait(('bet', entry.to_db_row()))
            except asyncio.QueueFull:
                logger.warning("bet_log_queue_full", dropping=entry.signal)

        logger.info(
            "bet_resolved_logged",
            signal=entry.signal,
            won=won,
            pnl=round(pnl, 4),
            balance=round(new_balance, 4),
        )

    @property
    def recent_bets(self) -> list[BetLogEntry]:
        return list(self._recent)

    @property
    def total_logged(self) -> int:
        return self._total_logged
