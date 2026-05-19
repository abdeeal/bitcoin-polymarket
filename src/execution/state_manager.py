"""
State Manager — Bot state machine for ACTIVE / COOLING_DOWN transitions.

State Machine (PRD 9.2):
  ┌──────────┐    2 consecutive losses    ┌──────────────┐
  │  ACTIVE  │ ─────────────────────────> │ COOLING_DOWN │
  │          │ <───────────────────────── │              │
  └──────────┘    elapsed >= 1800s        └──────────────┘

  BOOT → ACTIVE: on successful warm-up
  ACTIVE → COOLING_DOWN: triggered by LossTracker
  COOLING_DOWN → ACTIVE: triggered by expiry check (30 min elapsed)

Behaviour During Cooling Down (PRD 9.4):
  ✅ WebSocket feeds        — fully active
  ✅ Ring Buffer writes     — continuous
  ✅ Feature engine         — computes & persists every 5m
  ✅ Model inference        — runs (logged only, not executed)
  ✅ Signal generator       — generates (logged only)
  ❌ Bet execution          — blocked by bot_state_gate
  ❌ Compounding engine     — no balance changes
  ✅ Cooling down timer     — counting down

Design:
  StateManager is the single source of truth for BotState.
  All state transitions go through set_state() which:
    1. Validates the transition is legal
    2. Records transition metadata (timestamp, reason)
    3. Fires alert callbacks
    4. Logs to cooling_down_log history
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import structlog

from src.signal.signal_generator import BotState

logger = structlog.get_logger(__name__)

# Cooling down duration in seconds (PRD 9.2)
COOLING_DOWN_DURATION_SEC: float = 1800.0   # 30 minutes


@dataclass
class CoolingDownLogEntry:
    """Record of one cooling down period."""
    started_at: float
    ended_at: Optional[float] = None
    reason: str = ''
    duration_sec: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def to_dict(self) -> dict:
        return {
            'started_at': self.started_at,
            'ended_at': self.ended_at,
            'reason': self.reason,
            'duration_sec': round(self.duration_sec, 1),
            'is_active': self.is_active,
        }


class StateManager:
    """
    Central bot state machine.

    Owns BotState transitions and cooling down timer logic.
    Called by:
      - LossTracker.on_bet_resolved() → set_state(COOLING_DOWN)
      - Main loop tick → check_cooling_down_expiry()
      - Boot sequence → set_state(ACTIVE)
    """

    def __init__(
        self,
        initial_state: BotState = BotState.BOOT,
        cooling_down_duration: float = COOLING_DOWN_DURATION_SEC,
        on_alert: Optional[Callable[[str], None]] = None,
    ):
        self._state: BotState = initial_state
        self._cooling_down_duration = cooling_down_duration
        self._cooling_down_started_at: float = 0.0
        self._on_alert = on_alert

        # Cooling down history
        self._log: list[CoolingDownLogEntry] = []
        self._current_entry: Optional[CoolingDownLogEntry] = None

        logger.info("state_manager_init", state=self._state.name)

    # ────────────────────────────────────────────────────────────────────
    # State transitions
    # ────────────────────────────────────────────────────────────────────

    def set_state(self, new_state: BotState, reason: str = '') -> None:
        """
        Transition to a new state.

        Legal transitions:
          BOOT → ACTIVE
          ACTIVE → COOLING_DOWN
          COOLING_DOWN → ACTIVE
          Any → PAUSED (manual override)

        Args:
            new_state: Target state.
            reason:    Human-readable reason for the transition.
        """
        old_state = self._state

        if new_state == BotState.COOLING_DOWN:
            self._enter_cooling_down(reason)
        elif new_state == BotState.ACTIVE and old_state == BotState.COOLING_DOWN:
            self._exit_cooling_down()
        else:
            self._state = new_state

        logger.info(
            "state_transition",
            old=old_state.name,
            new=self._state.name,
            reason=reason or None,
        )

    def _enter_cooling_down(self, reason: str) -> None:
        """
        Enter cooling down state.

        Records the start time and creates a log entry.
        Fires alert callback with cooling down message.
        """
        self._state = BotState.COOLING_DOWN
        self._cooling_down_started_at = time.time()

        entry = CoolingDownLogEntry(
            started_at=self._cooling_down_started_at,
            reason=reason,
        )
        self._current_entry = entry
        self._log.append(entry)

        msg = f"🔴 COOLING DOWN — {reason}. {int(self._cooling_down_duration / 60)} min pause."
        logger.warning("cooling_down_started", reason=reason)
        if self._on_alert:
            self._on_alert(msg)

    def _exit_cooling_down(self) -> None:
        """
        Exit cooling down state back to ACTIVE.

        Records the end time in the log entry.
        Fires alert callback with resume message.
        """
        now = time.time()
        self._state = BotState.ACTIVE

        if self._current_entry:
            self._current_entry.ended_at = now
            self._current_entry.duration_sec = now - self._current_entry.started_at
            self._current_entry = None

        msg = "🟢 Cooling down ended — bot resumed."
        logger.info("cooling_down_ended")
        if self._on_alert:
            self._on_alert(msg)

    # ────────────────────────────────────────────────────────────────────
    # Timer check (called every tick by main loop)
    # ────────────────────────────────────────────────────────────────────

    def check_cooling_down_expiry(self) -> bool:
        """
        Check if cooling down period has expired.

        Called on every main loop tick (every 5-min candle).
        If elapsed >= 1800 seconds → transition back to ACTIVE.

        Returns:
            True if state transitioned from COOLING_DOWN to ACTIVE.
        """
        if self._state != BotState.COOLING_DOWN:
            return False

        elapsed = time.time() - self._cooling_down_started_at
        remaining = self._cooling_down_duration - elapsed

        if elapsed >= self._cooling_down_duration:
            self._exit_cooling_down()
            return True

        logger.debug(
            "cooling_down_active",
            elapsed_sec=round(elapsed, 0),
            remaining_sec=round(max(0, remaining), 0),
        )
        return False

    # ────────────────────────────────────────────────────────────────────
    # Properties
    # ────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == BotState.ACTIVE

    @property
    def is_cooling_down(self) -> bool:
        return self._state == BotState.COOLING_DOWN

    @property
    def cooling_down_remaining_sec(self) -> float:
        """Seconds remaining in cooling down. 0 if not cooling down."""
        if not self.is_cooling_down:
            return 0.0
        elapsed = time.time() - self._cooling_down_started_at
        return max(0.0, self._cooling_down_duration - elapsed)

    @property
    def cooling_down_log(self) -> list[CoolingDownLogEntry]:
        return self._log

    @property
    def total_cooling_downs(self) -> int:
        return len(self._log)

    def stats(self) -> dict:
        return {
            'state': self._state.name,
            'is_cooling_down': self.is_cooling_down,
            'cooling_down_remaining_sec': round(self.cooling_down_remaining_sec, 0),
            'total_cooling_downs': self.total_cooling_downs,
            'log': [e.to_dict() for e in self._log[-5:]],  # last 5 entries
        }
