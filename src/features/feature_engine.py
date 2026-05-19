"""
Feature Engine — Async orchestrator running in CPU thread pool.

Architecture Rules:
  - Reads ONLY from Ring Buffer (in-memory) — NEVER from DB
  - Runs in concurrent.futures.ThreadPoolExecutor (separate from I/O event loop)
  - Always executes regardless of bot state (ACTIVE, COOLING_DOWN, BOOT)
  - Pre-compiles all Numba JIT kernels at init
  - Total feature computation P95 < 100ms

Thread Pool Design:
  - ThreadPoolExecutor(max_workers=2)
  - Feature computation is CPU-bound but not parallelizable (sequential reads)
  - 2 workers: 1 for active computation, 1 for overlap during slow cycles
  - asyncio.run_in_executor bridges async I/O loop to sync CPU thread
"""

import asyncio
import concurrent.futures
import time
from typing import Optional

import numpy as np
import structlog

from src.memory.ring_buffer import RingBuffer
from src.features.jit_kernels import precompile_all_kernels
from src.features.price_action import compute_price_action
from src.features.volume import compute_volume_features
from src.features.technical import compute_technical
from src.features.order_book import compute_order_book
from src.features.microstructure import compute_microstructure
from src.features.cross_asset import compute_cross_asset
from src.features.sentiment import compute_sentiment
from src.features.calendar import compute_calendar
from src.features.lagged_interaction import compute_lagged_interaction

logger = structlog.get_logger(__name__)

# Thread pool configuration
THREAD_POOL_WORKERS = 2  # 1 active + 1 overlap
WINDOW_SIZE = 200         # Bars to read from Ring Buffer


class FeatureEngine:
    """
    Async feature computation engine running in a CPU thread pool.

    Computes 150+ features per 5-minute cycle from the Ring Buffer.
    Bridges async I/O event loop to synchronous CPU computation via
    concurrent.futures.ThreadPoolExecutor.

    Thread Safety:
      - Ring Buffer reads are thread-safe (RLock in get_window)
      - Feature computation is pure function (no shared mutable state)
      - Executor is created at init and shared across all compute calls
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        max_workers: int = THREAD_POOL_WORKERS,
        window_size: int = WINDOW_SIZE,
    ):
        """
        Initialize the feature engine.

        Args:
            ring_buffer: Shared Ring Buffer instance.
            max_workers: Thread pool size.
            window_size: Number of bars to read from Ring Buffer.
        """
        self._buf = ring_buffer
        self._window_size = window_size

        # Create thread pool for CPU-bound feature computation
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feature_engine",
        )

        # Performance tracking
        self._compute_count: int = 0
        self._total_latency_ms: float = 0
        self._last_latency_ms: float = 0
        self._max_latency_ms: float = 0

        # Pre-compile JIT kernels at init
        self._precompile_jit_kernels()

        logger.info(
            "feature_engine_initialized",
            max_workers=max_workers,
            window_size=window_size,
        )

    def _precompile_jit_kernels(self) -> None:
        """
        Warm up all Numba JIT kernels at init.

        This eliminates first-call JIT compilation latency (200ms-2s per kernel).
        Called once at boot — all subsequent calls use cached machine code.
        """
        t0 = time.perf_counter()
        try:
            precompile_all_kernels()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "jit_kernels_precompiled",
                elapsed_ms=round(elapsed_ms, 1),
            )
        except Exception as e:
            logger.warning(
                "jit_precompilation_failed",
                error=str(e),
                note="Will compile on first use (slower first cycle)",
            )

    async def compute_all(self, bot_state: str = "ACTIVE") -> dict:
        """
        Compute all 150+ features from the in-memory Ring Buffer.

        Runs in thread pool to avoid blocking the asyncio event loop.
        Always executes regardless of bot_state — feature continuity
        is required even during COOLING_DOWN to prevent data gaps.

        Args:
            bot_state: Current bot state ("ACTIVE", "COOLING_DOWN", "BOOT").

        Returns:
            dict of feature_name -> scalar value.
        """
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self._executor,
            self._compute_sync,
            bot_state,
        )
        return features

    def _compute_sync(self, bot_state: str) -> dict:
        """
        Synchronous feature computation — runs in CPU thread pool.

        Flow:
          1. Single read from Ring Buffer (get_window)
          2. Compute all feature categories sequentially
          3. Merge results into a single flat dict
          4. Track performance metrics
        """
        t0 = time.perf_counter()

        # 1. Single read from Ring Buffer — all fields, last N bars
        w = self._buf.get_window(self._window_size)

        if len(w.get('close', [])) < 5:
            logger.warning(
                "feature_engine_insufficient_data",
                available=len(w.get('close', [])),
                required=5,
            )
            return {'bot_state': bot_state, '_insufficient_data': True}

        # 2. Compute all feature categories
        features = {}

        # Price action (20 features, <5ms)
        features.update(compute_price_action(w))

        # Volume (15 features, <5ms)
        features.update(compute_volume_features(w))

        # Technical indicators (35 features, <20ms)
        features.update(compute_technical(w))

        # Order book (20 features, <15ms)
        features.update(compute_order_book(w))

        # Microstructure (15 features, <20ms)
        features.update(compute_microstructure(w))

        # Cross-asset (12 features, <10ms)
        features.update(compute_cross_asset(w))

        # Sentiment & derivatives (10 features, <10ms)
        features.update(compute_sentiment(w))

        # Calendar (8 features, <1ms)
        features.update(compute_calendar())

        # Lagged & interaction (15 features, <10ms)
        features.update(compute_lagged_interaction(w, features))

        # Add metadata
        features['bot_state'] = bot_state
        features['timestamp'] = int(w['timestamp'][-1]) if len(w['timestamp']) > 0 else 0
        features['buffer_size'] = self._buf.size

        # 3. Track performance
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._compute_count += 1
        self._total_latency_ms += elapsed_ms
        self._last_latency_ms = elapsed_ms
        self._max_latency_ms = max(self._max_latency_ms, elapsed_ms)

        logger.debug(
            "features_computed",
            feature_count=len(features),
            latency_ms=round(elapsed_ms, 2),
            bot_state=bot_state,
            buffer_size=self._buf.size,
        )

        return features

    def compute_sync(self, bot_state: str = "ACTIVE") -> dict:
        """
        Direct synchronous computation (for testing/backtest).

        Same as compute_all but without async wrapper.
        """
        return self._compute_sync(bot_state)

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)
        logger.info("feature_engine_shutdown")

    def stats(self) -> dict:
        """Return engine performance statistics."""
        avg = (
            self._total_latency_ms / max(self._compute_count, 1)
        )
        return {
            'compute_count': self._compute_count,
            'avg_latency_ms': round(avg, 2),
            'last_latency_ms': round(self._last_latency_ms, 2),
            'max_latency_ms': round(self._max_latency_ms, 2),
            'window_size': self._window_size,
            'buffer_size': self._buf.size,
        }
