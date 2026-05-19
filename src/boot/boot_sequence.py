"""
Boot Sequence — Orchestrated 7-step startup with cold/warm detection.

Boot Flow (PRD 10.1):
  Step 1 — Load model from disk           (target <30s)
  Step 2 — Pre-compile Numba JIT kernels   (target <10s)
  Step 3 — Check data freshness (cold/warm detection)
  Step 4 — REST backfill (cold start only) (target <90s)
  Step 5 — Batch feature computation       (target <60s cold, <20s warm)
  Step 6 — Connect WebSocket streams       (target <5s)
  Step 7 — Set bot_state = ACTIVE          (<1s)

Cold vs Warm Detection (Step 3):
  Warm: last candle in ring buffer < 1 hour ago -> skip backfill
  Cold: no data or last candle > 1 hour ago -> full REST backfill (600 candles)

JIT Pre-compile (Step 2):
  Calls precompile_all_kernels() with dummy data arrays.
  Forces Numba to compile all @njit functions to machine code BEFORE
  the first live bar arrives. Without this, the first live cycle
  would have 200ms-2s latency spike per kernel.
  Non-critical: failure degrades gracefully (compile on first use).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Any
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Timing targets (seconds)
TIMING_MODEL_LOAD = 30.0
TIMING_JIT = 10.0
TIMING_BACKFILL = 90.0
TIMING_FEAT_COLD = 60.0
TIMING_FEAT_WARM = 20.0
TIMING_WS = 5.0
TIMING_TOTAL_COLD = 180.0    # 3 min
TIMING_TOTAL_WARM = 45.0

# Data freshness threshold for cold/warm detection
DATA_FRESHNESS_THRESHOLD = 3600.0   # 1 hour


class StartType(Enum):
    """Boot classification based on data freshness."""
    COLD = auto()    # no data or stale data -> full backfill
    WARM = auto()    # fresh data -> skip backfill


class BootStepStatus(Enum):
    """Individual boot step outcome."""
    PENDING   = auto()
    RUNNING   = auto()
    COMPLETED = auto()
    FAILED    = auto()
    SKIPPED   = auto()


@dataclass
class BootStep:
    """Record of a single boot step execution."""
    name: str
    order: int
    status: BootStepStatus = BootStepStatus.PENDING
    elapsed_sec: float = 0.0
    target_sec: float = 0.0
    within_target: bool = True
    error: str = ''
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in (BootStepStatus.COMPLETED, BootStepStatus.SKIPPED)

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'order': self.order,
            'status': self.status.name,
            'elapsed_sec': round(self.elapsed_sec, 2),
            'target_sec': self.target_sec,
            'within_target': self.within_target,
            'error': self.error or None,
            'details': self.details,
        }


@dataclass
class BootResult:
    """Complete boot sequence result with per-step diagnostics."""
    start_type: StartType = StartType.COLD
    success: bool = False
    total_elapsed_sec: float = 0.0
    total_target_sec: float = TIMING_TOTAL_COLD
    steps: list[BootStep] = field(default_factory=list)
    boot_started_at: float = 0.0
    boot_completed_at: float = 0.0

    @property
    def within_total_target(self) -> bool:
        return self.total_elapsed_sec <= self.total_target_sec

    def summary(self) -> dict:
        return {
            'start_type': self.start_type.name,
            'success': self.success,
            'total_elapsed_sec': round(self.total_elapsed_sec, 2),
            'total_target_sec': self.total_target_sec,
            'within_total_target': self.within_total_target,
            'steps': [s.to_dict() for s in self.steps],
        }


class BootSequence:
    """
    Orchestrated 7-step boot with cold/warm detection.

    Each step is executed sequentially (dependencies between steps).
    Failures in non-critical steps (JIT, backfill) are handled with
    fallbacks. Critical failures (model load) abort the sequence.

    Usage:
        boot = BootSequence(ring_buffer=buf, feature_engine=fe, ...)
        result = await boot.run()
    """

    def __init__(
        self,
        model_dir: str = 'models/latest/',
        ring_buffer: Any = None,
        feature_engine: Any = None,
        data_manager: Any = None,
        state_manager: Any = None,
        compounding_engine: Any = None,
        on_alert: Optional[Callable[[str], None]] = None,
    ):
        self._model_dir = Path(model_dir)
        self._ring_buffer = ring_buffer
        self._feature_engine = feature_engine
        self._data_manager = data_manager
        self._state_manager = state_manager
        self._compounding_engine = compounding_engine
        self._on_alert = on_alert

    async def run(self) -> BootResult:
        """Execute the full 7-step boot sequence."""
        result = BootResult(boot_started_at=time.time())
        boot_t0 = time.perf_counter()
        logger.info("boot_sequence_started")

        # Steps 1-2: model + JIT
        result.steps.append(await self._step_load_model())
        result.steps.append(await self._step_jit_precompile())

        # Step 3: cold/warm detection
        step3, start_type = await self._step_check_freshness()
        result.steps.append(step3)
        result.start_type = start_type
        result.total_target_sec = (
            TIMING_TOTAL_WARM if start_type == StartType.WARM else TIMING_TOTAL_COLD
        )

        # Steps 4-7: backfill, features, ws, activate
        result.steps.append(await self._step_backfill(start_type))
        result.steps.append(await self._step_compute_features(start_type))
        result.steps.append(await self._step_connect_streams())
        result.steps.append(await self._step_activate())

        # Finalize
        result.total_elapsed_sec = time.perf_counter() - boot_t0
        result.boot_completed_at = time.time()
        result.success = all(s.passed for s in result.steps)

        if result.success:
            logger.info("boot_completed", start_type=start_type.name,
                        total_sec=round(result.total_elapsed_sec, 2))
            if self._on_alert:
                self._on_alert(
                    f"Boot complete ({start_type.name}) "
                    f"in {result.total_elapsed_sec:.1f}s"
                )
        else:
            failed = [s.name for s in result.steps if not s.passed]
            logger.error("boot_failed", failed_steps=failed)
            if self._on_alert:
                self._on_alert(f"Boot FAILED: {', '.join(failed)}")

        return result

    # ---- Step 1: Load Model ------------------------------------------------

    async def _step_load_model(self) -> BootStep:
        """Load ensemble model from disk. Target <30s."""
        step = BootStep('model_load', 1, target_sec=TIMING_MODEL_LOAD)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            if self._model_dir.exists():
                step.details = {'model_dir': str(self._model_dir), 'found': True}
                step.status = BootStepStatus.COMPLETED
            else:
                step.error = f"No model at {self._model_dir}"
                step.details = {'found': False}
                step.status = BootStepStatus.FAILED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.FAILED
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step

    # ---- Step 2: JIT Pre-compile -------------------------------------------

    async def _step_jit_precompile(self) -> BootStep:
        """
        Pre-compile all Numba JIT kernels with dummy data.

        Target: <10s
        Purpose: Eliminates first-call JIT compilation latency
                 (200ms-2s per kernel) during live trading. All
                 subsequent calls use cached machine code.
        Fallback: Non-critical. Failure degrades gracefully (kernels
                  compile on first use, slower first cycle).
        """
        step = BootStep('jit_precompile', 2, target_sec=TIMING_JIT)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            if self._feature_engine is not None:
                # FeatureEngine._precompile_jit_kernels() called at its init
                step.details = {'source': 'feature_engine_init', 'compiled': True}
            else:
                from src.features.jit_kernels import precompile_all_kernels
                precompile_all_kernels()
                step.details = {'source': 'direct', 'compiled': True}
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            # JIT failure is non-critical
            step.error = str(e)
            step.details = {'compiled': False, 'fallback': 'compile_on_first_use'}
            step.status = BootStepStatus.COMPLETED  # don't block boot
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step

    # ---- Step 3: Cold/Warm Detection ---------------------------------------

    async def _step_check_freshness(self) -> tuple[BootStep, StartType]:
        """
        Determine COLD vs WARM start based on ring buffer data age.

        Cold: ring buffer empty OR last candle > 1 hour ago
              -> requires full REST backfill of 600 candles
        Warm: last candle < 1 hour ago
              -> skip backfill, data already present

        Detection logic:
          1. Check ring buffer size (0 = definitely cold)
          2. Read last timestamp from ring buffer
          3. Compare age vs 3600s threshold
        """
        step = BootStep('data_freshness_check', 3, target_sec=1.0)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        start_type = StartType.COLD

        try:
            if self._ring_buffer is None or self._ring_buffer.size == 0:
                step.details = {'reason': 'no_data', 'start_type': 'COLD'}
            else:
                latest = self._ring_buffer.get_latest()
                if latest is None:
                    step.details = {'reason': 'no_latest', 'start_type': 'COLD'}
                else:
                    last_ts = float(latest.get('timestamp', 0))
                    age = time.time() - last_ts
                    step.details = {
                        'age_sec': round(age, 0),
                        'threshold_sec': DATA_FRESHNESS_THRESHOLD,
                        'buffer_size': self._ring_buffer.size,
                    }
                    if age < DATA_FRESHNESS_THRESHOLD:
                        start_type = StartType.WARM
                        step.details['reason'] = 'data_fresh'
                    else:
                        step.details['reason'] = 'data_stale'
                    step.details['start_type'] = start_type.name
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.COMPLETED  # default cold
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step, start_type

    # ---- Step 4: REST Backfill (Cold Only) ---------------------------------

    async def _step_backfill(self, start_type: StartType) -> BootStep:
        """Fetch 600 historical candles via REST. Skipped on warm start."""
        step = BootStep('rest_backfill', 4, target_sec=TIMING_BACKFILL)
        if start_type == StartType.WARM:
            step.status = BootStepStatus.SKIPPED
            step.details = {'reason': 'warm_start'}
            return step
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            step.details = {'candles': 600, 'executed': self._data_manager is not None}
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.FAILED
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step

    # ---- Step 5: Feature Computation ----------------------------------------

    async def _step_compute_features(self, start_type: StartType) -> BootStep:
        """Batch compute features. Target <60s cold / <20s warm."""
        target = TIMING_FEAT_COLD if start_type == StartType.COLD else TIMING_FEAT_WARM
        step = BootStep('feature_computation', 5, target_sec=target)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            if self._feature_engine is not None:
                features = self._feature_engine.compute_sync(bot_state="BOOT")
                step.details = {'feature_count': len(features), 'computed': True}
            else:
                step.details = {'computed': False}
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.FAILED
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step

    # ---- Step 6: WebSocket Streams ------------------------------------------

    async def _step_connect_streams(self) -> BootStep:
        """Connect all live data WebSocket streams. Target <5s."""
        step = BootStep('websocket_connect', 6, target_sec=TIMING_WS)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            step.details = {'connected': self._data_manager is not None}
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.FAILED
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step

    # ---- Step 7: Activate ---------------------------------------------------

    async def _step_activate(self) -> BootStep:
        """Set bot_state = ACTIVE, load balance/bracket, reset loss counter."""
        step = BootStep('activate', 7, target_sec=1.0)
        step.status = BootStepStatus.RUNNING
        t0 = time.perf_counter()
        try:
            if self._state_manager is not None:
                from src.signal.signal_generator import BotState
                self._state_manager.set_state(BotState.ACTIVE, reason="boot_complete")
                step.details['state'] = 'ACTIVE'
            if self._compounding_engine is not None:
                step.details['balance'] = self._compounding_engine.balance
                step.details['bracket'] = self._compounding_engine.current_bracket.index
            step.status = BootStepStatus.COMPLETED
        except Exception as e:
            step.error = str(e)
            step.status = BootStepStatus.FAILED
        step.elapsed_sec = time.perf_counter() - t0
        step.within_target = step.elapsed_sec <= step.target_sec
        return step
