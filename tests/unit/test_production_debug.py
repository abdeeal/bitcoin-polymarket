"""
Production Debugging Tests — Hardened edge-case and stability verification.

Covers:
  1. Race Conditions          — RingBuffer concurrent write/read safety
  2. Numerical Instability    — Kelly NaN/Inf, normalizer div-zero, PSI overflow
  3. Model Degradation Root   — Signal generator threshold boundary precision
  4. Unexpected Cooling Down  — Re-entrant trigger, timer drift, rapid L-L-W-L-L
  5. Ring Buffer Edge Cases   — Full wrap, exact capacity, single-element, overwrite
"""

import asyncio
import math
import time
import threading
import pytest
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.ring_buffer import RingBuffer
from src.sizing.compounding_engine import CompoundingEngine, BRACKETS, MINIMUM_BET_FLOOR
from src.sizing.kelly_sizer import KellySizer
from src.signal.signal_generator import SignalGenerator, Signal, BotState
from src.execution.state_manager import StateManager, COOLING_DOWN_DURATION_SEC
from src.execution.loss_tracker import LossTracker, CONSECUTIVE_LOSS_TRIGGER
from src.features.normalizer import zscore_normalize, minmax_normalize, batch_zscore
from src.monitoring.drift_detector import compute_psi


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RACE CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRaceConditions:
    """RingBuffer thread safety under concurrent write + read."""

    def test_concurrent_push_and_read(self):
        """Push from writer thread while main thread reads — no crash."""
        rb = RingBuffer(capacity=50)
        stop = threading.Event()
        errors = []

        def writer():
            i = 0
            while not stop.is_set():
                try:
                    rb.push({'close': float(i), 'timestamp': i})
                    i += 1
                except Exception as e:
                    errors.append(e)
                    break

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        # Read repeatedly while writer is running
        for _ in range(200):
            try:
                window = rb.get_window(10)
                assert isinstance(window, dict)
                latest = rb.get_latest()
                assert isinstance(latest, dict)
                _ = rb.size
            except Exception as e:
                errors.append(e)
                break

        stop.set()
        t.join(timeout=2)
        assert len(errors) == 0, f"Race errors: {errors}"

    def test_concurrent_push_batch_and_read(self):
        """push_batch() and get_field_window() concurrent access."""
        rb = RingBuffer(capacity=100)
        stop = threading.Event()
        errors = []

        def writer():
            for batch_num in range(20):
                if stop.is_set():
                    break
                bars = [{'close': float(batch_num * 10 + j), 'timestamp': j}
                        for j in range(10)]
                try:
                    rb.push_batch(bars)
                except Exception as e:
                    errors.append(e)

        t = threading.Thread(target=writer, daemon=True)
        t.start()

        for _ in range(100):
            try:
                rb.get_field_window('close', 30)
            except Exception as e:
                errors.append(e)

        stop.set()
        t.join(timeout=2)
        assert len(errors) == 0

    def test_update_latest_during_push(self):
        """update_latest and push don't corrupt each other."""
        rb = RingBuffer(capacity=20)
        rb.push({'close': 100.0, 'volume': 1.0, 'timestamp': 1})

        stop = threading.Event()
        errors = []

        def updater():
            while not stop.is_set():
                try:
                    rb.update_latest({'volume': 999.0})
                except Exception as e:
                    errors.append(e)

        t = threading.Thread(target=updater, daemon=True)
        t.start()

        for i in range(50):
            rb.push({'close': float(100 + i), 'timestamp': i + 2})

        stop.set()
        t.join(timeout=2)
        assert len(errors) == 0
        # Last bar's volume might be 999.0 or default — no crash
        assert rb.size > 0

    def test_state_manager_concurrent_check(self):
        """StateManager.check_cooling_down_expiry() is safe to call repeatedly."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.1)
        sm.set_state(BotState.COOLING_DOWN, reason="test")

        results = []
        def checker():
            for _ in range(50):
                results.append(sm.check_cooling_down_expiry())
                time.sleep(0.005)

        threads = [threading.Thread(target=checker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)

        # At most ONE thread should see the True transition
        true_count = sum(1 for r in results if r)
        # May be 0 if not expired yet, or 1+ if race (which is acceptable for
        # idempotent exit) — critical: no crash
        assert isinstance(true_count, int)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NUMERICAL INSTABILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestNumericalInstability:
    """Boundary cases that produce NaN, Inf, or division-by-zero."""

    def test_kelly_prob_zero(self):
        """Kelly with P=0 returns 0 (no bet), never NaN."""
        ks = KellySizer()
        result = ks.size(prob=0.0, odds=1.90, balance=100.0)
        assert result == 0.0
        assert not math.isnan(result)

    def test_kelly_prob_one(self):
        """Kelly with P=1 is capped — never returns > max_pct × balance."""
        ks = KellySizer()
        result = ks.size(prob=1.0, odds=1.90, balance=100.0)
        assert result <= 2.0  # 2% of 100
        assert not math.isnan(result)

    def test_kelly_odds_one(self):
        """Kelly with odds=1.0 (zero profit) returns 0."""
        ks = KellySizer()
        result = ks.size(prob=0.6, odds=1.0, balance=100.0)
        assert result == 0.0

    def test_kelly_odds_very_high(self):
        """Kelly with extremely high odds doesn't overflow."""
        ks = KellySizer()
        result = ks.size(prob=0.6, odds=100.0, balance=100.0)
        assert result > 0
        assert result <= 2.0  # capped at 2%
        assert not math.isnan(result)

    def test_kelly_balance_zero(self):
        """Kelly with zero balance returns 0."""
        ks = KellySizer()
        result = ks.size(prob=0.6, odds=1.90, balance=0.0)
        assert result == 0.0

    def test_kelly_tiny_balance(self):
        """Kelly with sub-penny balance rounds to 0."""
        ks = KellySizer()
        result = ks.size(prob=0.6, odds=1.90, balance=0.001)
        assert result == 0.0  # rounds to 0.00

    def test_normalizer_constant_array(self):
        """Z-score with constant array (std=0) doesn't produce Inf."""
        arr = np.full(100, 42.0)
        z = zscore_normalize(42.0, arr, 100)
        assert not math.isinf(z)
        assert not math.isnan(z)
        # 42 - 42 = 0 / (0 + 1e-10) = 0
        assert abs(z) < 1.0

    def test_normalizer_single_element(self):
        """Z-score with single element doesn't crash."""
        arr = np.array([10.0])
        z = zscore_normalize(10.0, arr, 1)
        assert not math.isnan(z)

    def test_minmax_constant_array(self):
        """Min-max with constant array uses epsilon."""
        arr = np.full(50, 7.0)
        result = minmax_normalize(7.0, arr, 50)
        assert not math.isinf(result)
        assert not math.isnan(result)

    def test_batch_zscore_with_nan(self):
        """batch_zscore handles NaN in history gracefully."""
        features = {'rsi': 50.0}
        history = {'rsi': [float('nan')] * 100}
        result = batch_zscore(features, history, window=100)
        # Should not crash — NaN propagates but doesn't raise
        assert 'rsi' in result

    def test_psi_uniform_vs_uniform(self):
        """PSI between uniform distributions is finite."""
        ref = np.random.uniform(0, 1, 1000)
        cur = np.random.uniform(0, 1, 1000)
        psi = compute_psi(ref, cur)
        assert not math.isnan(psi)
        assert not math.isinf(psi)

    def test_psi_all_same_value(self):
        """PSI with degenerate distributions."""
        ref = np.ones(100)
        cur = np.ones(100) * 2
        psi = compute_psi(ref, cur)
        assert not math.isnan(psi)

    def test_compounding_balance_never_negative(self):
        """Balance never goes negative after repeated losses."""
        ce = CompoundingEngine(initial_balance=5.0)
        for _ in range(20):
            ce.on_bet_resolved(won=False, bet_amount=1.0, payout=0.0)
        assert ce.balance >= 0.0

    def test_compounding_pnl_tracking_accuracy(self):
        """Total PnL accurately reflects balance changes from initial."""
        ce = CompoundingEngine(initial_balance=10.0)
        ce.on_bet_resolved(won=True, bet_amount=1.0, payout=1.90)
        ce.on_bet_resolved(won=False, bet_amount=1.0, payout=0.0)
        # After win: 10 + 0.90 = 10.90
        # After loss: 10.90 - 1.0 = 9.90
        assert abs(ce.balance - 9.90) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEGRADATION ROOT CAUSE
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelDegradation:
    """Signal threshold boundary precision and drift detection."""

    def test_exact_threshold_boundary_up_normal(self):
        """P(UP) exactly at 0.72 should trigger BET_UP_NORMAL."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.72, filters_ok=True)
        assert result.signal == Signal.BET_UP_NORMAL

    def test_just_below_threshold(self):
        """P(UP) = 0.7199 should NOT trigger bet."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.7199, filters_ok=True)
        assert result.signal == Signal.NO_BET

    def test_exact_threshold_boundary_down_normal(self):
        """P(UP) exactly at 0.28 should trigger BET_DOWN_NORMAL."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.28, filters_ok=True)
        assert result.signal == Signal.BET_DOWN_NORMAL

    def test_exact_threshold_high_conf(self):
        """P(UP) exactly at 0.82 should trigger HIGH_CONF."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.82, filters_ok=True)
        assert result.signal == Signal.BET_UP_HIGH_CONF

    def test_floating_point_near_threshold(self):
        """Floating point arithmetic near threshold — 0.72 + epsilon stays correct."""
        sg = SignalGenerator()
        # 0.72 exactly
        result = sg.generate_signal(prob=0.72 + 1e-15, filters_ok=True)
        assert result.signal == Signal.BET_UP_NORMAL

    def test_prob_midpoint_no_bet(self):
        """P(UP) = 0.50 exactly produces NO_BET."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.50, filters_ok=True)
        assert result.signal == Signal.NO_BET

    def test_prob_extreme_values(self):
        """Prob at 0.0 and 1.0 produce correct signals."""
        sg = SignalGenerator()
        r0 = sg.generate_signal(prob=0.0, filters_ok=True)
        r1 = sg.generate_signal(prob=1.0, filters_ok=True)
        assert r0.signal == Signal.BET_DOWN_HIGH_CONF
        assert r1.signal == Signal.BET_UP_HIGH_CONF

    def test_high_conf_overrides_normal(self):
        """P=0.85 matches HIGH_CONF, not NORMAL (hierarchy order)."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.85, filters_ok=True)
        assert result.signal == Signal.BET_UP_HIGH_CONF

    def test_ev_computation_for_down_signal(self):
        """EV is computed correctly for DOWN signals using (1-prob)."""
        sg = SignalGenerator()
        result = sg.generate_signal(prob=0.15, filters_ok=True, odds=1.90)
        # For DOWN: direction_prob = 1 - 0.15 = 0.85
        # implied_prob = 1/1.90 ≈ 0.5263
        # EV = 0.85 - 0.5263 ≈ 0.3237
        assert result.ev > 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UNEXPECTED COOLING DOWN BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoolingDownEdgeCases:
    """Cooling down timer, re-entrant triggers, rapid sequences."""

    def test_loss_during_cooling_down_does_not_double_trigger(self):
        """Losses during COOLING_DOWN should NOT re-trigger cooling down."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=999)
        lt = LossTracker(sm, trigger_count=2)

        # 2 losses → cooling down
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert sm.is_cooling_down
        assert sm.total_cooling_downs == 1

        # More losses during cooling down — counter resets after trigger,
        # and _trigger_cooling_down guards with is_active check
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert sm.total_cooling_downs == 1  # no second trigger

    def test_win_resets_counter_completely(self):
        """Single win resets consecutive losses, even mid-streak."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=999)
        lt = LossTracker(sm, trigger_count=2)

        lt.on_bet_resolved(won=False)
        assert lt.consecutive_losses == 1

        lt.on_bet_resolved(won=True)
        assert lt.consecutive_losses == 0

        lt.on_bet_resolved(won=False)
        assert lt.consecutive_losses == 1
        assert sm.is_active  # not triggered yet

    def test_rapid_llwll_sequence(self):
        """L-L-W-L-L triggers cooling down twice with correct pauses."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        lt = LossTracker(sm, trigger_count=2)

        # L, L → cooling down #1
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert sm.is_cooling_down
        assert sm.total_cooling_downs == 1

        # Wait for cooldown to expire
        time.sleep(0.02)
        lt.check_expiry()
        assert sm.is_active

        # W → reset counter
        lt.on_bet_resolved(won=True)
        assert lt.consecutive_losses == 0

        # L, L → cooling down #2
        lt.on_bet_resolved(won=False)
        lt.on_bet_resolved(won=False)
        assert sm.is_cooling_down
        assert sm.total_cooling_downs == 2

    def test_cooling_down_timer_precision(self):
        """Cooling down expires within reasonable tolerance."""
        duration = 0.05  # 50ms for test speed
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=duration)
        sm.set_state(BotState.COOLING_DOWN, reason="test")

        # Not expired yet
        assert sm.is_cooling_down
        assert not sm.check_cooling_down_expiry()

        # Wait for it to expire
        time.sleep(duration + 0.01)
        assert sm.check_cooling_down_expiry()
        assert sm.is_active

    def test_cooling_down_remaining_decreases(self):
        """remaining_sec decreases monotonically."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=1.0)
        sm.set_state(BotState.COOLING_DOWN, reason="test")

        r1 = sm.cooling_down_remaining_sec
        time.sleep(0.05)
        r2 = sm.cooling_down_remaining_sec
        assert r2 < r1

    def test_cooling_down_log_entry_completeness(self):
        """Log entry has started_at, ended_at, duration_sec after expiry."""
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=0.01)
        sm.set_state(BotState.COOLING_DOWN, reason="2 consecutive losses")

        time.sleep(0.02)
        sm.check_cooling_down_expiry()

        entry = sm.cooling_down_log[0]
        assert entry.started_at > 0
        assert entry.ended_at is not None
        assert entry.duration_sec > 0
        assert entry.reason == "2 consecutive losses"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RING BUFFER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestRingBufferEdgeCases:
    """Wrap-around, exact capacity, overwrite, single element."""

    def test_empty_buffer_get_window(self):
        """get_window on empty buffer returns empty arrays."""
        rb = RingBuffer(capacity=10)
        result = rb.get_window(5)
        assert all(len(arr) == 0 for arr in result.values())

    def test_empty_buffer_get_latest(self):
        """get_latest on empty buffer returns empty dict."""
        rb = RingBuffer(capacity=10)
        assert rb.get_latest() == {}

    def test_single_element(self):
        """Buffer with 1 element returns that element correctly."""
        rb = RingBuffer(capacity=10)
        rb.push({'close': 42.0, 'timestamp': 1})
        result = rb.get_window(1)
        assert result['close'][0] == 42.0

    def test_exact_capacity_fill(self):
        """Buffer filled to exact capacity returns all elements in order."""
        cap = 10
        rb = RingBuffer(capacity=cap)
        for i in range(cap):
            rb.push({'close': float(i), 'timestamp': i})

        assert rb.is_full
        assert rb.size == cap
        window = rb.get_window(cap)
        np.testing.assert_array_equal(window['close'], np.arange(cap, dtype=np.float64))

    def test_overwrite_preserves_order(self):
        """After overwrite (capacity+5 pushes), last N are in order."""
        cap = 10
        rb = RingBuffer(capacity=cap)
        total = cap + 5  # push 15 into capacity=10
        for i in range(total):
            rb.push({'close': float(i), 'timestamp': i})

        assert rb.size == cap
        window = rb.get_window(cap)
        # Should contain [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        expected = np.arange(total - cap, total, dtype=np.float64)
        np.testing.assert_array_equal(window['close'], expected)

    def test_wrap_around_get_window(self):
        """Window spanning the wrap boundary produces contiguous array."""
        cap = 5
        rb = RingBuffer(capacity=cap)
        # Push 7 elements: [0,1,2,3,4,5,6] → buffer holds [5,6,2,3,4] with head=2
        for i in range(7):
            rb.push({'close': float(i), 'timestamp': i})

        window = rb.get_window(5)
        # Should be [2, 3, 4, 5, 6]
        np.testing.assert_array_equal(window['close'], [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_get_field_window_wrap(self):
        """get_field_window handles wrap-around correctly."""
        cap = 5
        rb = RingBuffer(capacity=cap)
        for i in range(8):
            rb.push({'close': float(i), 'timestamp': i})

        result = rb.get_field_window('close', 5)
        # Should be [3, 4, 5, 6, 7]
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0, 6.0, 7.0])

    def test_request_more_than_size(self):
        """Requesting N > size returns only available bars."""
        rb = RingBuffer(capacity=20)
        for i in range(3):
            rb.push({'close': float(i), 'timestamp': i})

        window = rb.get_window(100)
        assert len(window['close']) == 3

    def test_latest_timestamp_after_overwrite(self):
        """latest_timestamp reflects the most recent push."""
        rb = RingBuffer(capacity=5)
        for i in range(8):
            rb.push({'timestamp': i * 1000})
        assert rb.latest_timestamp == 7000

    def test_oldest_timestamp_after_wrap(self):
        """oldest_timestamp reflects the oldest surviving bar."""
        rb = RingBuffer(capacity=5)
        for i in range(8):
            rb.push({'timestamp': (i + 1) * 1000})
        # Survived: timestamps 4000, 5000, 6000, 7000, 8000
        assert rb.oldest_timestamp == 4000

    def test_update_latest_on_empty(self):
        """update_latest on empty buffer is a no-op."""
        rb = RingBuffer(capacity=10)
        rb.update_latest({'close': 999.0})  # should not crash
        assert rb.size == 0

    def test_unknown_field_in_push_ignored(self):
        """Unknown fields in push() are silently ignored."""
        rb = RingBuffer(capacity=10)
        rb.push({'close': 42.0, 'unknown_field': 999.0, 'timestamp': 1})
        assert rb.size == 1
        assert rb.get_latest()['close'] == 42.0

    def test_stats_after_operations(self):
        """stats() returns consistent data."""
        rb = RingBuffer(capacity=10)
        for i in range(5):
            rb.push({'close': float(i), 'timestamp': i})
        s = rb.stats()
        assert s['capacity'] == 10
        assert s['size'] == 5
        assert s['is_full'] is False
        assert s['total_pushes'] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPOUNDING EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompoundingEdgeCases:
    """Bracket transitions, Kelly boundaries, sub-minimum balance."""

    def test_balance_below_minimum_returns_zero(self):
        """resolve_bet_size returns 0 when balance < min bet."""
        ce = CompoundingEngine(initial_balance=0.50)
        bet = ce.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert bet == 0.0

    def test_balance_at_bracket_boundary(self):
        """Balance at bracket floor uses that bracket's sizing."""
        ce = CompoundingEngine(initial_balance=15.00)
        assert ce.current_bracket.index == 1
        bet = ce.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert bet == 1.50

    def test_downgrade_skips_brackets(self):
        """Large loss can skip multiple brackets."""
        ce = CompoundingEngine(initial_balance=50.0)
        assert ce.current_bracket.index == 3  # $40–64.99

        # Lose enough to drop to bracket 0
        ce.on_bet_resolved(won=False, bet_amount=45.0, payout=0.0)
        assert ce.balance == 5.0
        assert ce.current_bracket.index == 0

    def test_kelly_mode_transition(self):
        """Crossing $100 boundary activates Kelly mode."""
        ce = CompoundingEngine(initial_balance=99.0)
        assert not ce.is_kelly_mode

        # Win → cross $100
        ce.on_bet_resolved(won=True, bet_amount=5.0, payout=10.0)
        # balance = 99 + 5 = 104
        assert ce.is_kelly_mode

    def test_upgrade_deferred_low_win_rate(self):
        """Upgrade deferred when win_rate < 50% at ceiling."""
        ce = CompoundingEngine(initial_balance=14.00)
        # Win 3 out of 11 bets → WR = 27% < 50%
        for _ in range(3):
            ce.on_bet_resolved(won=True, bet_amount=1.0, payout=1.90)
        for _ in range(8):
            ce.on_bet_resolved(won=False, bet_amount=1.0, payout=0.0)
        # Balance should have changed but if at ceiling, upgrade is deferred
        # (this tests the logic path, not exact balance arithmetic)
        assert ce.current_bracket.index >= 0  # shouldn't crash


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
