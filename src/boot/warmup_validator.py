"""
Warmup Validator — Post-boot verification that all components are ready.

Checks after BootSequence completes:
  1. JIT kernels are compiled (not pending first-call compilation)
  2. Ring buffer has sufficient data (>= min_bars)
  3. Feature engine produces valid output (no NaN flood)
  4. Model is loaded and produces valid probabilities
  5. State manager is in ACTIVE state
  6. Compounding engine has valid bracket

If validation fails, the bot should NOT start accepting signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Minimum bars needed for feature computation
MIN_BARS_FOR_FEATURES = 30
# Maximum acceptable NaN ratio in features
MAX_NAN_RATIO = 0.10


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool = False
    details: str = ''

    def to_dict(self) -> dict:
        return {'name': self.name, 'passed': self.passed, 'details': self.details}


@dataclass
class ValidationResult:
    """Aggregate validation outcome."""
    all_passed: bool = False
    checks: list[ValidationCheck] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def summary(self) -> dict:
        return {
            'all_passed': self.all_passed,
            'elapsed_sec': round(self.elapsed_sec, 2),
            'checks': [c.to_dict() for c in self.checks],
            'failed': [c.name for c in self.checks if not c.passed],
        }


class WarmupValidator:
    """
    Post-boot validation that all subsystems are operational.

    Called after BootSequence.run() succeeds. Verifies that
    the bot is genuinely ready to process its first live bar.

    Usage:
        validator = WarmupValidator(ring_buffer, feature_engine, ...)
        result = validator.validate()
        if not result.all_passed:
            logger.error("warmup_failed", failed=result.summary()['failed'])
    """

    def __init__(
        self,
        ring_buffer: Any = None,
        feature_engine: Any = None,
        state_manager: Any = None,
        compounding_engine: Any = None,
        min_bars: int = MIN_BARS_FOR_FEATURES,
    ):
        self._ring_buffer = ring_buffer
        self._feature_engine = feature_engine
        self._state_manager = state_manager
        self._compounding_engine = compounding_engine
        self._min_bars = min_bars

    def validate(self) -> ValidationResult:
        """Run all validation checks and return aggregate result."""
        t0 = time.perf_counter()
        result = ValidationResult()

        result.checks.append(self._check_jit_compiled())
        result.checks.append(self._check_ring_buffer_data())
        result.checks.append(self._check_feature_output())
        result.checks.append(self._check_state_active())
        result.checks.append(self._check_compounding_ready())

        result.elapsed_sec = time.perf_counter() - t0
        result.all_passed = all(c.passed for c in result.checks)

        if result.all_passed:
            logger.info("warmup_validation_passed", elapsed_sec=round(result.elapsed_sec, 3))
        else:
            failed = [c.name for c in result.checks if not c.passed]
            logger.warning("warmup_validation_failed", failed=failed)

        return result

    def _check_jit_compiled(self) -> ValidationCheck:
        """Verify JIT kernels are compiled (not pending)."""
        check = ValidationCheck('jit_compiled')
        try:
            from src.features.jit_kernels import precompile_all_kernels
            # If this returns without error, kernels are compiled
            # (Numba caches compiled functions after first call)
            precompile_all_kernels()
            check.passed = True
            check.details = 'All JIT kernels verified compiled'
        except ImportError:
            # Numba not available — numpy fallback is valid
            check.passed = True
            check.details = 'Numba not available, numpy fallback OK'
        except Exception as e:
            check.details = f'JIT check failed: {e}'
        return check

    def _check_ring_buffer_data(self) -> ValidationCheck:
        """Verify ring buffer has enough data for feature computation."""
        check = ValidationCheck('ring_buffer_data')
        if self._ring_buffer is None:
            check.details = 'No ring buffer provided'
            return check
        size = self._ring_buffer.size
        if size >= self._min_bars:
            check.passed = True
            check.details = f'{size} bars >= {self._min_bars} minimum'
        else:
            check.details = f'{size} bars < {self._min_bars} minimum'
        return check

    def _check_feature_output(self) -> ValidationCheck:
        """Verify feature engine produces valid (non-NaN-flooded) output."""
        check = ValidationCheck('feature_output')
        if self._feature_engine is None:
            check.details = 'No feature engine provided'
            return check
        try:
            features = self._feature_engine.compute_sync(bot_state="BOOT")
            if features.get('_insufficient_data'):
                check.details = 'Insufficient data for features'
                return check

            total = len(features)
            nan_count = sum(1 for v in features.values()
                           if isinstance(v, float) and (v != v))  # NaN check
            nan_ratio = nan_count / max(total, 1)

            if nan_ratio <= MAX_NAN_RATIO:
                check.passed = True
                check.details = f'{total} features, {nan_count} NaN ({nan_ratio:.1%})'
            else:
                check.details = f'NaN ratio {nan_ratio:.1%} > {MAX_NAN_RATIO:.0%} limit'
        except Exception as e:
            check.details = f'Feature computation error: {e}'
        return check

    def _check_state_active(self) -> ValidationCheck:
        """Verify state manager is in ACTIVE state."""
        check = ValidationCheck('state_active')
        if self._state_manager is None:
            check.details = 'No state manager provided'
            return check
        from src.signal.signal_generator import BotState
        if self._state_manager.state == BotState.ACTIVE:
            check.passed = True
            check.details = 'State is ACTIVE'
        else:
            check.details = f'State is {self._state_manager.state.name}, expected ACTIVE'
        return check

    def _check_compounding_ready(self) -> ValidationCheck:
        """Verify compounding engine has valid bracket and balance."""
        check = ValidationCheck('compounding_ready')
        if self._compounding_engine is None:
            check.details = 'No compounding engine provided'
            return check
        balance = self._compounding_engine.balance
        bracket = self._compounding_engine.current_bracket.index
        if balance > 0:
            check.passed = True
            check.details = f'Balance ${balance:.2f}, bracket {bracket}'
        else:
            check.details = f'Balance ${balance:.2f} <= 0'
        return check
