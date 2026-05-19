"""
Tests for Phase 6 — Cooling Down System.

Covers:
  1. State machine transitions (BOOT→ACTIVE, ACTIVE→COOLING_DOWN, COOLING_DOWN→ACTIVE)
  2. Consecutive loss counter (win resets, loss increments)
  3. Cooling down trigger (2 consecutive losses → COOLING_DOWN)
  4. Cooling down expiry (elapsed >= 1800s → ACTIVE)
  5. Counter reset on trigger (no accumulation across cooling periods)
  6. Behaviour matrix: features run, bets blocked during cooling down
  7. Alert callbacks fired on state transitions
  8. Cooling down log history
  9. Integration: LossTracker + StateManager + RiskGates(bot_state_gate)
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.signal.signal_generator import BotState
from src.execution.state_manager import StateManager, COOLING_DOWN_DURATION_SEC
from src.execution.loss_tracker import LossTracker, CONSECUTIVE_LOSS_TRIGGER


# ═══════════════════════════════════════════════════════════════════════════════
# State Manager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateManagerTransitions:
    """Test legal state transitions."""

    def test_initial_state_boot(self):
        sm = StateManager(initial_state=BotState.BOOT)
        assert sm.state == BotState.BOOT

    def test_boot_to_active(self):
        sm = StateManager(initial_state=BotState.BOOT)
        sm.set_state(BotState.ACTIVE, reason="warm-up complete")
        assert sm.is_active

    def test_active_to_cooling_down(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        sm.set_state(BotState.COOLING_DOWN, reason="2 consecutive losses")
        assert sm.is_cooling_down

    def test_cooling_down_to_active(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        sm.set_state(BotState.ACTIVE, reason="manual resume")
        assert sm.is_active

    def test_any_to_paused(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        sm.set_state(BotState.PAUSED, reason="manual stop")
        assert sm.state == BotState.PAUSED


class TestCoolingDownTimer:
    """Test 30-minute cooling down timer and expiry."""

    def test_cooling_down_duration_constant(self):
        assert COOLING_DOWN_DURATION_SEC == 1800.0

    def test_not_expired_immediately(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=10.0)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        expired = sm.check_cooling_down_expiry()
        assert not expired
        assert sm.is_cooling_down

    def test_expired_after_duration(self):
        """Use a very short duration to test expiry."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        time.sleep(0.02)  # wait past duration
        expired = sm.check_cooling_down_expiry()
        assert expired
        assert sm.is_active

    def test_expiry_check_when_not_cooling_down(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        expired = sm.check_cooling_down_expiry()
        assert not expired

    def test_cooling_down_remaining_sec(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=100.0)
        assert sm.cooling_down_remaining_sec == 0.0
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        remaining = sm.cooling_down_remaining_sec
        assert 98.0 <= remaining <= 100.0

    def test_cooling_down_remaining_zero_after_expiry(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        time.sleep(0.02)
        sm.check_cooling_down_expiry()
        assert sm.cooling_down_remaining_sec == 0.0


class TestCoolingDownLog:
    """Test cooling down log history."""

    def test_log_entry_created_on_enter(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        sm.set_state(BotState.COOLING_DOWN, reason="2 losses")
        assert sm.total_cooling_downs == 1
        entry = sm.cooling_down_log[0]
        assert entry.is_active
        assert entry.reason == "2 losses"

    def test_log_entry_completed_on_exit(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        time.sleep(0.02)
        sm.check_cooling_down_expiry()
        entry = sm.cooling_down_log[0]
        assert not entry.is_active
        assert entry.ended_at is not None
        assert entry.duration_sec > 0

    def test_multiple_cooling_down_periods_logged(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        for i in range(3):
            sm.set_state(BotState.COOLING_DOWN, reason=f"round {i}")
            time.sleep(0.02)
            sm.check_cooling_down_expiry()
        assert sm.total_cooling_downs == 3


class TestStateManagerAlerts:
    """Test alert callbacks fire on transitions."""

    def test_alert_on_cooling_down_start(self):
        alerts = []
        sm = StateManager(initial_state=BotState.ACTIVE, on_alert=alerts.append)
        sm.set_state(BotState.COOLING_DOWN, reason="2 losses")
        assert any('COOLING DOWN' in a for a in alerts)

    def test_alert_on_cooling_down_end(self):
        alerts = []
        sm = StateManager(
            initial_state=BotState.ACTIVE,
            cooling_down_duration=0.01,
            on_alert=alerts.append,
        )
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        time.sleep(0.02)
        sm.check_cooling_down_expiry()
        assert any('resumed' in a for a in alerts)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss Tracker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsecutiveLossCounter:
    """PRD 9.3 — consecutive loss counter logic."""

    def _make_tracker(self, **sm_kwargs) -> tuple[LossTracker, StateManager]:
        sm = StateManager(initial_state=BotState.ACTIVE, **sm_kwargs)
        lt = LossTracker(sm)
        return lt, sm

    def test_trigger_constant(self):
        assert CONSECUTIVE_LOSS_TRIGGER == 2

    def test_win_resets_counter(self):
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=False)    # consecutive=1
        lt.on_bet_resolved(won=True)     # consecutive=0
        assert lt.consecutive_losses == 0

    def test_single_loss_no_trigger(self):
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=False)
        assert lt.consecutive_losses == 1
        assert sm.is_active

    def test_two_losses_trigger_cooling_down(self):
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=False)   # 1
        lt.on_bet_resolved(won=False)   # 2 → trigger
        assert sm.is_cooling_down

    def test_counter_reset_on_trigger(self):
        """Counter resets to 0 after triggering cooling down."""
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)   # trigger
        assert lt.consecutive_losses == 0  # reset

    def test_win_loss_win_loss_no_trigger(self):
        """W L W L pattern never reaches 2 consecutive."""
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=True)
        lt.on_bet_resolved(won=False)   # 1
        lt.on_bet_resolved(won=True)    # 0
        lt.on_bet_resolved(won=False)   # 1
        assert sm.is_active
        assert lt.consecutive_losses == 1

    def test_three_losses_trigger_once(self):
        """Third loss comes after reset — only triggers once at the second loss."""
        lt, sm = self._make_tracker(cooling_down_duration=0.01)
        lt.on_bet_resolved(won=False)   # 1
        lt.on_bet_resolved(won=False)   # 2 → trigger, reset to 0
        time.sleep(0.02)
        lt.check_expiry()               # back to ACTIVE
        lt.on_bet_resolved(won=False)   # 1 (fresh start)
        assert sm.is_active             # not triggered yet (only 1 loss)

    def test_max_streak_tracked(self):
        lt, sm = self._make_tracker()
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)   # streak=2, triggers
        assert lt.stats()['max_streak'] == 2


class TestLossTrackerAlerts:
    """Test Telegram alert callbacks."""

    def test_alert_on_trigger(self):
        alerts = []
        sm = StateManager(initial_state=BotState.ACTIVE)
        lt = LossTracker(sm, on_alert=alerts.append)
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert any('COOLING DOWN' in a for a in alerts)

    def test_alert_on_resume(self):
        alerts = []
        sm = StateManager(
            initial_state=BotState.ACTIVE,
            cooling_down_duration=0.01,
        )
        lt = LossTracker(sm, on_alert=alerts.append)
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)   # trigger
        time.sleep(0.02)
        lt.check_expiry()
        assert any('resumed' in a for a in alerts)


class TestLossTrackerStats:
    def test_stats_structure(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        lt = LossTracker(sm)
        lt.on_bet_resolved(won=True)
        lt.on_bet_resolved(won=False)
        s = lt.stats()
        assert 'consecutive_losses' in s
        assert 'total_resolved' in s
        assert 'total_triggers' in s
        assert s['total_resolved'] == 2
        assert s['total_losses'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoolingDownIntegration:
    """End-to-end: LossTracker → StateManager → bot_state_gate."""

    def test_full_cooling_down_cycle(self):
        """
        Scenario: win, lose, lose → cooling down → 30min → active
        """
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        lt = LossTracker(sm)

        # Win
        lt.on_bet_resolved(won=True)
        assert sm.is_active

        # Loss 1
        lt.on_bet_resolved(won=False)
        assert sm.is_active

        # Loss 2 → trigger
        lt.on_bet_resolved(won=False)
        assert sm.is_cooling_down

        # During cooling down, feature engine still runs (verified by PRD 9.4)
        # ... but bet execution is blocked

        # Wait for expiry
        time.sleep(0.02)
        lt.check_expiry()
        assert sm.is_active

    def test_bot_state_gate_blocks_during_cooling_down(self):
        """Risk gate 6 blocks bets during cooling down."""
        from src.signal.risk_gates import _gate_bot_state

        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        lt = LossTracker(sm)

        # Before cooling down — gate passes
        passed, _ = _gate_bot_state(sm.state)
        assert passed

        # Trigger cooling down
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)

        # During cooling down — gate blocks
        passed, reason = _gate_bot_state(sm.state)
        assert not passed
        assert 'BOT_STATE' in reason

        # After expiry — gate passes again
        time.sleep(0.02)
        lt.check_expiry()
        passed, _ = _gate_bot_state(sm.state)
        assert passed

    def test_feature_collection_continues(self):
        """
        Verify the contract: during cooling down, the feature engine
        can still be called. This test verifies the state manager
        does not block anything other than state checks.
        """
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        lt = LossTracker(sm)

        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert sm.is_cooling_down

        # During cooling down: model inference and feature computation continue
        # (they don't check bot_state — only bet execution checks it)
        # This is a design contract test — the state_manager does NOT prevent
        # any operations other than those that explicitly check sm.state
        assert sm.state == BotState.COOLING_DOWN

        # Signal generator generates signals (logged only), this is OK
        from src.signal.signal_generator import SignalGenerator, Signal
        gen = SignalGenerator()
        result = gen.generate_signal(0.85, filters_ok=True, bot_state=sm.state)
        assert result.signal == Signal.NO_BET   # blocked by bot_state

        # But the signal WAS computed — just not executed
        # The signal generator produces NO_BET, which the execution layer
        # interprets as "log but don't execute"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
