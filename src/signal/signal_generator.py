"""
Signal Generator — Threshold logic and signal tiering.

PRD 7.1: Two-tier confidence system.

Threshold Design (tuned for BTC 5-min market efficiency):
  ┌────────────────────────────────────────────────────────────────┐
  │  P(UP) >= 0.62 → BET_UP_HIGH_CONF   ($2 bet, ~3-5% candles) │
  │  P(UP) >= 0.55 → BET_UP_NORMAL      ($1 bet, ~10-15% candles)│
  │  P(UP) <= 0.45 → BET_DOWN_NORMAL    ($1 bet, ~10-15% candles)│
  │  P(UP) <= 0.38 → BET_DOWN_HIGH_CONF ($2 bet, ~3-5% candles) │
  │  else          → NO_BET                                       │
  └────────────────────────────────────────────────────────────────┘

Key Design Decisions:
  - HIGH_CONF signals have NO frequency cap — consecutive signals all fire.
  - NORMAL signals also have no cap (cooling down is outcome-based, not
    frequency-based per PRD).
  - Cooling down is triggered by 2 consecutive LOSSES, not signal count.
  - Thresholds are constants — not tuned per market session.

Signal Flow:
  P(UP) from EnsembleModel
    → generate_signal(prob, filters_ok=risk_gates.all_pass(features, odds))
    → Signal enum
    → CompoundingEngine.resolve_bet_size(signal)
    → Execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time

import structlog

logger = structlog.get_logger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class Signal(Enum):
    """All possible signal outputs from the generator."""
    NO_BET             = auto()   # filtered or uncertainty zone
    BET_UP_NORMAL      = auto()   # P >= 0.55, $1 bet
    BET_DOWN_NORMAL    = auto()   # P <= 0.45, $1 bet
    BET_UP_HIGH_CONF   = auto()   # P >= 0.62, $2 bet, no freq cap
    BET_DOWN_HIGH_CONF = auto()   # P <= 0.38, $2 bet, no freq cap

    @property
    def is_bet(self) -> bool:
        return self != Signal.NO_BET

    @property
    def is_up(self) -> bool:
        return self in (Signal.BET_UP_NORMAL, Signal.BET_UP_HIGH_CONF)

    @property
    def is_down(self) -> bool:
        return self in (Signal.BET_DOWN_NORMAL, Signal.BET_DOWN_HIGH_CONF)

    @property
    def is_high_conf(self) -> bool:
        return self in (Signal.BET_UP_HIGH_CONF, Signal.BET_DOWN_HIGH_CONF)

    @property
    def direction(self) -> str:
        if self.is_up:
            return 'UP'
        if self.is_down:
            return 'DOWN'
        return 'NONE'


class BotState(Enum):
    """Bot operational state — controls whether bets execute."""
    BOOT         = auto()   # starting up, no bets
    ACTIVE       = auto()   # normal operation, bets execute
    COOLING_DOWN = auto()   # 2 consecutive losses — bets blocked for 30 min
    PAUSED       = auto()   # hard stop triggered (balance < $5.00)


# ─── Signal metadata ──────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Rich signal result with full context for logging and execution."""
    signal: Signal
    prob: float
    bot_state: BotState
    timestamp: float = field(default_factory=time.time)

    # Gate information
    gates_passed: bool = True
    gate_failures: list[str] = field(default_factory=list)

    # Odds information
    odds: float = 0.0
    implied_prob: float = 0.0
    ev: float = 0.0                # expected value edge = model_prob - implied_prob

    # Thresholds (for transparency)
    threshold_used: float = 0.0

    def to_dict(self) -> dict:
        return {
            'signal': self.signal.name,
            'prob': round(self.prob, 4),
            'direction': self.signal.direction,
            'is_high_conf': self.signal.is_high_conf,
            'bot_state': self.bot_state.name,
            'gates_passed': self.gates_passed,
            'gate_failures': self.gate_failures,
            'odds': round(self.odds, 3),
            'implied_prob': round(self.implied_prob, 4),
            'ev': round(self.ev, 4),
            'threshold_used': self.threshold_used,
            'timestamp': self.timestamp,
        }


# ─── Signal Generator ─────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Two-tier confidence signal generator with threshold-based classification.

    Thresholds (tuned for BTC 5-min market efficiency):
      NORMAL tier:    P >= 0.55 (UP) or P <= 0.45 (DOWN)   → $1 bet
      HIGH_CONF tier: P >= 0.62 (UP) or P <= 0.38 (DOWN)   → $2 bet

    Frequency Policy:
      - NO frequency cap on any tier
      - HIGH_CONF consecutive signals ALL fire
      - Cooling down triggered by 2 consecutive LOSS OUTCOMES, not signal count
    """

    # ── Threshold constants (PRD-defined, not tunable per session) ──
    THRESHOLD_UP_NORMAL: float   = 0.55
    THRESHOLD_DOWN_NORMAL: float = 0.45
    THRESHOLD_UP_HIGH: float     = 0.60
    THRESHOLD_DOWN_HIGH: float   = 0.40

    def __init__(self):
        self._signal_count: int = 0
        self._last_signal: Signal = Signal.NO_BET
        self._last_signal_time: float = 0.0

    def generate_signal(
        self,
        prob: float,
        filters_ok: bool,
        bot_state: BotState = BotState.ACTIVE,
        odds: float = 0.0,
        gate_failures: Optional[list[str]] = None,
    ) -> SignalResult:
        """
        Classify probability into a signal tier.

        PRD Rule: filters_ok=False → NO_BET regardless of probability.
        HIGH_CONF signals fire consecutively without cap.

        Args:
            prob:          Calibrated P(UP) from ensemble, range [0, 1].
            filters_ok:    True only if ALL 6 risk gates pass.
            bot_state:     Current bot state (ACTIVE, COOLING_DOWN, etc.).
            odds:          Polymarket decimal odds for EV computation.
            gate_failures: List of gate names that failed (for logging).

        Returns:
            SignalResult with signal, metadata, and gate context.
        """
        failures = gate_failures or []

        # Blocked: any gate failed or bot not active
        if not filters_ok or bot_state != BotState.ACTIVE:
            result = SignalResult(
                signal=Signal.NO_BET,
                prob=prob,
                bot_state=bot_state,
                gates_passed=False,
                gate_failures=failures,
                odds=odds,
            )
            logger.debug(
                "signal_blocked",
                prob=round(prob, 4),
                bot_state=bot_state.name,
                gate_failures=failures,
            )
            return result

        # ── Threshold classification (HIGH_CONF checked first) ──
        signal, threshold = self._classify(prob)

        # Compute EV if odds provided
        implied_prob = (1.0 / odds) if odds > 0 else 0.0
        ev = prob - implied_prob if signal.is_up else (1.0 - prob) - implied_prob

        result = SignalResult(
            signal=signal,
            prob=prob,
            bot_state=bot_state,
            gates_passed=True,
            gate_failures=[],
            odds=odds,
            implied_prob=implied_prob,
            ev=ev,
            threshold_used=threshold,
        )

        if signal.is_bet:
            self._signal_count += 1
            self._last_signal = signal
            self._last_signal_time = result.timestamp

            logger.info(
                "signal_generated",
                signal=signal.name,
                prob=round(prob, 4),
                direction=signal.direction,
                high_conf=signal.is_high_conf,
                ev=round(ev, 4),
                threshold=threshold,
                consecutive_ok=True,  # no cap — always fires
            )
        else:
            logger.debug("signal_no_bet", prob=round(prob, 4))

        return result

    def _classify(self, prob: float) -> tuple[Signal, float]:
        """
        Apply threshold hierarchy and return (Signal, threshold_used).

        HIGH_CONF checked first — a prob of 0.85 matches HIGH_CONF,
        not NORMAL (even though it also exceeds the NORMAL threshold).
        """
        if prob >= self.THRESHOLD_UP_HIGH:
            return Signal.BET_UP_HIGH_CONF, self.THRESHOLD_UP_HIGH
        if prob <= self.THRESHOLD_DOWN_HIGH:
            return Signal.BET_DOWN_HIGH_CONF, self.THRESHOLD_DOWN_HIGH
        if prob >= self.THRESHOLD_UP_NORMAL:
            return Signal.BET_UP_NORMAL, self.THRESHOLD_UP_NORMAL
        if prob <= self.THRESHOLD_DOWN_NORMAL:
            return Signal.BET_DOWN_NORMAL, self.THRESHOLD_DOWN_NORMAL
        return Signal.NO_BET, 0.0

    def stats(self) -> dict:
        """Return signal generation statistics."""
        return {
            'total_signals': self._signal_count,
            'last_signal': self._last_signal.name,
            'last_signal_time': self._last_signal_time,
            'thresholds': {
                'up_normal': self.THRESHOLD_UP_NORMAL,
                'down_normal': self.THRESHOLD_DOWN_NORMAL,
                'up_high': self.THRESHOLD_UP_HIGH,
                'down_high': self.THRESHOLD_DOWN_HIGH,
            },
        }
