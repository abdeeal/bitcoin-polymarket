"""
Tests for Phase 7 — Fast Warm-Up & Boot Sequence.

Covers:
  1. Boot sequence orchestration (all 7 steps)
  2. Cold vs warm detection logic
  3. JIT pre-compile timing and fallback
  4. Step-level timing and target tracking
  5. Backfill skipped on warm start
  6. Activate sets ACTIVE state
  7. BootResult summary diagnostics
  8. WarmupValidator post-boot checks
"""

import pytest
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.boot.boot_sequence import (
    BootSequence, BootResult, BootStep, BootStepStatus,
    StartType, DATA_FRESHNESS_THRESHOLD,
    TIMING_TOTAL_COLD, TIMING_TOTAL_WARM,
)
from src.boot.warmup_validator import WarmupValidator, ValidationResult
from src.signal.signal_generator import BotState
from src.execution.state_manager import StateManager
from src.sizing.compounding_engine import CompoundingEngine


# ---- Helpers ---------------------------------------------------------------

class FakeRingBuffer:
    """Minimal ring buffer mock for boot tests."""
    def __init__(self, size: int = 0, last_ts: float = 0.0):
        self._size = size
        self._last_ts = last_ts

    @property
    def size(self) -> int:
        return self._size

    def get_latest(self):
        if self._size == 0:
            return None
        return {'timestamp': self._last_ts}


class FakeFeatureEngine:
    """Minimal feature engine mock."""
    def __init__(self, feature_count: int = 155):
        self._feature_count = feature_count
        self._jit_compiled = True

    def compute_sync(self, bot_state: str = "BOOT") -> dict:
        return {f'feature_{i}': float(i) for i in range(self._feature_count)}


def run_async(coro):
    """Run async coroutine in sync test."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# Boot Sequence Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootSequenceOrchestration:
    """Test the 7-step boot flow."""

    def test_boot_produces_7_steps(self):
        """All 7 steps are executed and recorded."""
        sm = StateManager(initial_state=BotState.BOOT)
        ce = CompoundingEngine(10.0)
        boot = BootSequence(
            model_dir='.',  # current dir exists
            state_manager=sm,
            compounding_engine=ce,
        )
        result = run_async(boot.run())
        assert len(result.steps) == 7

    def test_step_order_1_to_7(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        orders = [s.order for s in result.steps]
        assert orders == [1, 2, 3, 4, 5, 6, 7]

    def test_step_names(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        names = [s.name for s in result.steps]
        assert 'model_load' in names
        assert 'jit_precompile' in names
        assert 'data_freshness_check' in names
        assert 'rest_backfill' in names
        assert 'feature_computation' in names
        assert 'websocket_connect' in names
        assert 'activate' in names

    def test_all_steps_have_timing(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        for step in result.steps:
            assert step.elapsed_sec >= 0

    def test_boot_success_when_model_exists(self):
        sm = StateManager(initial_state=BotState.BOOT)
        boot = BootSequence(model_dir='.', state_manager=sm)
        result = run_async(boot.run())
        assert result.success

    def test_boot_fails_when_model_missing(self):
        boot = BootSequence(model_dir='nonexistent_model_path_xyz')
        result = run_async(boot.run())
        assert not result.success
        model_step = result.steps[0]
        assert model_step.status == BootStepStatus.FAILED

    def test_boot_result_summary(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        s = result.summary()
        assert 'start_type' in s
        assert 'success' in s
        assert 'total_elapsed_sec' in s
        assert 'steps' in s
        assert len(s['steps']) == 7


class TestBootAlerts:
    def test_success_alert(self):
        alerts = []
        sm = StateManager(initial_state=BotState.BOOT)
        boot = BootSequence(model_dir='.', state_manager=sm, on_alert=alerts.append)
        run_async(boot.run())
        assert any('Boot complete' in a for a in alerts)

    def test_failure_alert(self):
        alerts = []
        boot = BootSequence(model_dir='nonexistent_xyz', on_alert=alerts.append)
        run_async(boot.run())
        assert any('FAILED' in a for a in alerts)


# ═══════════════════════════════════════════════════════════════════════════════
# Cold vs Warm Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestColdWarmDetection:
    """Test data freshness detection logic."""

    def test_no_ring_buffer_is_cold(self):
        boot = BootSequence(model_dir='.', ring_buffer=None)
        result = run_async(boot.run())
        assert result.start_type == StartType.COLD

    def test_empty_ring_buffer_is_cold(self):
        buf = FakeRingBuffer(size=0)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        assert result.start_type == StartType.COLD

    def test_stale_data_is_cold(self):
        """Last candle > 1 hour ago -> COLD start."""
        stale_ts = time.time() - 7200  # 2 hours ago
        buf = FakeRingBuffer(size=100, last_ts=stale_ts)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        assert result.start_type == StartType.COLD

    def test_fresh_data_is_warm(self):
        """Last candle < 1 hour ago -> WARM start."""
        fresh_ts = time.time() - 300  # 5 minutes ago
        buf = FakeRingBuffer(size=100, last_ts=fresh_ts)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        assert result.start_type == StartType.WARM

    def test_exactly_1_hour_is_cold(self):
        """Age >= threshold -> COLD (not strictly less than)."""
        ts = time.time() - DATA_FRESHNESS_THRESHOLD
        buf = FakeRingBuffer(size=100, last_ts=ts)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        assert result.start_type == StartType.COLD

    def test_warm_start_total_target(self):
        fresh_ts = time.time() - 60
        buf = FakeRingBuffer(size=100, last_ts=fresh_ts)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        assert result.total_target_sec == TIMING_TOTAL_WARM

    def test_cold_start_total_target(self):
        boot = BootSequence(model_dir='.', ring_buffer=None)
        result = run_async(boot.run())
        assert result.total_target_sec == TIMING_TOTAL_COLD

    def test_freshness_step_details(self):
        fresh_ts = time.time() - 60
        buf = FakeRingBuffer(size=100, last_ts=fresh_ts)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        step3 = result.steps[2]
        assert step3.name == 'data_freshness_check'
        assert 'age_sec' in step3.details
        assert step3.details['start_type'] == 'WARM'


# ═══════════════════════════════════════════════════════════════════════════════
# Backfill Behavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackfillBehavior:
    """Test that backfill is skipped on warm start."""

    def test_cold_start_runs_backfill(self):
        boot = BootSequence(model_dir='.', ring_buffer=None)
        result = run_async(boot.run())
        backfill_step = result.steps[3]
        assert backfill_step.name == 'rest_backfill'
        assert backfill_step.status != BootStepStatus.SKIPPED

    def test_warm_start_skips_backfill(self):
        buf = FakeRingBuffer(size=100, last_ts=time.time() - 60)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        backfill_step = result.steps[3]
        assert backfill_step.status == BootStepStatus.SKIPPED
        assert backfill_step.details.get('reason') == 'warm_start'

    def test_skipped_step_passes(self):
        buf = FakeRingBuffer(size=100, last_ts=time.time() - 60)
        boot = BootSequence(model_dir='.', ring_buffer=buf)
        result = run_async(boot.run())
        backfill_step = result.steps[3]
        assert backfill_step.passed  # SKIPPED counts as passed


# ═══════════════════════════════════════════════════════════════════════════════
# JIT Pre-compile Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestJITPrecompile:
    """Test JIT pre-compilation step."""

    def test_jit_step_completes(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        jit_step = result.steps[1]
        assert jit_step.name == 'jit_precompile'
        assert jit_step.passed

    def test_jit_with_feature_engine(self):
        fe = FakeFeatureEngine()
        boot = BootSequence(model_dir='.', feature_engine=fe)
        result = run_async(boot.run())
        jit_step = result.steps[1]
        assert jit_step.details.get('source') == 'feature_engine_init'

    def test_jit_failure_is_non_critical(self):
        """JIT failure should NOT block boot."""
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        jit_step = result.steps[1]
        # Whether JIT succeeds or fails with fallback, step should pass
        assert jit_step.passed

    def test_jit_has_timing(self):
        boot = BootSequence(model_dir='.')
        result = run_async(boot.run())
        jit_step = result.steps[1]
        assert jit_step.elapsed_sec >= 0
        assert jit_step.target_sec == 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# Activate Step Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestActivateStep:
    """Test Step 7 — bot activation."""

    def test_activate_sets_active_state(self):
        sm = StateManager(initial_state=BotState.BOOT)
        boot = BootSequence(model_dir='.', state_manager=sm)
        run_async(boot.run())
        assert sm.is_active

    def test_activate_records_balance(self):
        sm = StateManager(initial_state=BotState.BOOT)
        ce = CompoundingEngine(10.0)
        boot = BootSequence(model_dir='.', state_manager=sm, compounding_engine=ce)
        result = run_async(boot.run())
        activate_step = result.steps[6]
        assert activate_step.details.get('balance') == 10.0
        assert activate_step.details.get('bracket') == 0

    def test_boot_with_feature_engine(self):
        sm = StateManager(initial_state=BotState.BOOT)
        fe = FakeFeatureEngine()
        boot = BootSequence(model_dir='.', state_manager=sm, feature_engine=fe)
        result = run_async(boot.run())
        feat_step = result.steps[4]
        assert feat_step.details.get('computed') is True
        assert feat_step.details.get('feature_count') == 155


# ═══════════════════════════════════════════════════════════════════════════════
# Warmup Validator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarmupValidator:
    """Test post-boot validation checks."""

    def test_all_checks_pass(self):
        sm = StateManager(initial_state=BotState.ACTIVE)
        ce = CompoundingEngine(10.0)
        buf = FakeRingBuffer(size=100)
        fe = FakeFeatureEngine()
        v = WarmupValidator(ring_buffer=buf, feature_engine=fe,
                            state_manager=sm, compounding_engine=ce)
        result = v.validate()
        assert result.all_passed

    def test_ring_buffer_too_small_fails(self):
        buf = FakeRingBuffer(size=5)
        v = WarmupValidator(ring_buffer=buf, min_bars=30)
        result = v.validate()
        rb_check = [c for c in result.checks if c.name == 'ring_buffer_data'][0]
        assert not rb_check.passed

    def test_state_not_active_fails(self):
        sm = StateManager(initial_state=BotState.BOOT)
        v = WarmupValidator(state_manager=sm)
        result = v.validate()
        state_check = [c for c in result.checks if c.name == 'state_active'][0]
        assert not state_check.passed

    def test_zero_balance_fails(self):
        ce = CompoundingEngine(0.0)
        v = WarmupValidator(compounding_engine=ce)
        result = v.validate()
        comp_check = [c for c in result.checks if c.name == 'compounding_ready'][0]
        assert not comp_check.passed

    def test_validation_result_summary(self):
        v = WarmupValidator()
        result = v.validate()
        s = result.summary()
        assert 'all_passed' in s
        assert 'checks' in s
        assert 'failed' in s

    def test_validation_has_timing(self):
        v = WarmupValidator()
        result = v.validate()
        assert result.elapsed_sec >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# BootStep / BootResult Data Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootStepData:
    def test_boot_step_to_dict(self):
        step = BootStep('test', 1, status=BootStepStatus.COMPLETED,
                        elapsed_sec=0.5, target_sec=10.0)
        d = step.to_dict()
        assert d['name'] == 'test'
        assert d['status'] == 'COMPLETED'
        assert d['within_target'] is True

    def test_boot_step_passed_completed(self):
        step = BootStep('x', 1, status=BootStepStatus.COMPLETED)
        assert step.passed

    def test_boot_step_passed_skipped(self):
        step = BootStep('x', 1, status=BootStepStatus.SKIPPED)
        assert step.passed

    def test_boot_step_not_passed_failed(self):
        step = BootStep('x', 1, status=BootStepStatus.FAILED)
        assert not step.passed

    def test_start_type_enum(self):
        assert StartType.COLD.name == 'COLD'
        assert StartType.WARM.name == 'WARM'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
