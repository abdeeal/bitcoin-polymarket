"""
Loss Tracker — Consecutive loss counter and cooling down trigger.

PRD 9.3: Tracks consecutive losses and triggers cooling down
after 2 consecutive losses. Any win fully resets the counter.

State Machine:
  on_bet_resolved(won=True)  → consecutive_losses = 0 (any win resets)
  on_bet_resolved(won=False) → consecutive_losses += 1
    if consecutive_losses >= 2:
      → trigger cooling down (ACTIVE → COOLING_DOWN)
      → reset counter to 0

Counter never exceeds the trigger threshold because
it is reset upon triggering cooling down.

Integration:
  LossTracker owns the consecutive loss counter.
  StateManager owns the bot state and cooling down timer.
  LossTracker calls state_manager.set_state(COOLING_DOWN) when triggered.
"""

from __future__ import annotations

import time
from typing import Optional, Callable

import structlog

from src.execution.state_manager import StateManager
from src.signal.signal_generator import BotState

logger = structlog.get_logger(__name__)

# Number of consecutive losses to trigger cooling down
CONSECUTIVE_LOSS_TRIGGER: int = 2


class LossTracker:
    """
    Consecutive loss counter with cooling down trigger.

    Called by the execution layer after every bet resolution.
    Delegates state transitions to StateManager.
    """

    def __init__(
        self,
        state_manager: StateManager,
        trigger_count: int = CONSECUTIVE_LOSS_TRIGGER,
        on_alert: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            state_manager: Central state machine (owns BotState).
            trigger_count: Consecutive losses to trigger cooling down (default: 2).
            on_alert:      Callback for Telegram alerts.
        """
        self._state_manager = state_manager
        self._trigger_count = trigger_count
        self._on_alert = on_alert
        self._consecutive_losses: int = 0
        self._last_loss_time: float = 0.0

        # Lifetime stats
        self._total_resolved: int = 0
        self._total_losses: int = 0
        self._total_triggers: int = 0
        self._max_streak: int = 0

    def on_bet_resolved(self, won: bool) -> None:
        """
        Update consecutive loss counter after a bet resolves.

        Args:
            won: True if the bet was correct.
        """
        self._total_resolved += 1

        if won:
            self._consecutive_losses = 0
            logger.debug("loss_tracker_win_reset", consecutive=0)
        else:
            self._total_losses += 1
            self._consecutive_losses += 1
            self._last_loss_time = time.time()

            if self._consecutive_losses > self._max_streak:
                self._max_streak = self._consecutive_losses

            logger.info(
                "loss_tracker_loss",
                consecutive=self._consecutive_losses,
                trigger_at=self._trigger_count,
            )

            if self._consecutive_losses >= self._trigger_count:
                self._trigger_cooling_down()

    def _trigger_cooling_down(self) -> None:
        """
        Trigger cooling down via StateManager.

        Resets the consecutive loss counter to 0 (fresh start after
        cooling down ends). The counter does NOT accumulate across
        cooling down periods.
        """
        self._total_triggers += 1
        reason = f"{self._consecutive_losses} consecutive losses"
        self._consecutive_losses = 0   # reset on trigger
        self._last_loss_time = 0.0

        logger.warning(
            "cooling_down_triggered",
            reason=reason,
            total_triggers=self._total_triggers,
        )

        # Only trigger if currently ACTIVE (ignore if already cooling down)
        if self._state_manager.is_active:
            self._state_manager.set_state(BotState.COOLING_DOWN, reason=reason)

            if self._on_alert:
                self._on_alert(
                    f"🔴 COOLING DOWN: {reason}. "
                    f"30 min pause. Signals logged but not executed."
                )

    def check_expiry(self) -> bool:
        """
        Check if cooling down has expired (delegates to StateManager).

        Called every main loop tick. Returns True if state transitioned
        from COOLING_DOWN to ACTIVE.
        """
        transitioned = self._state_manager.check_cooling_down_expiry()

        if transitioned and self._on_alert:
            self._on_alert("🟢 Cooling down ended — bot resumed.")

        # Single loss expiration (30 mins without entry resets consecutive_losses)
        if self._consecutive_losses > 0 and self._last_loss_time > 0:
            if time.time() - self._last_loss_time >= 1800:  # 30 mins
                logger.info("loss_tracker_loss_expired", reason="30 mins without entry")
                self._consecutive_losses = 0
                self._last_loss_time = 0.0

        return transitioned

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def is_cooling_down(self) -> bool:
        return self._state_manager.is_cooling_down

    def stats(self) -> dict:
        return {
            'consecutive_losses': self._consecutive_losses,
            'trigger_count': self._trigger_count,
            'total_resolved': self._total_resolved,
            'total_losses': self._total_losses,
            'total_triggers': self._total_triggers,
            'max_streak': self._max_streak,
            'is_cooling_down': self.is_cooling_down,
            'state': self._state_manager.state.name,
        }
