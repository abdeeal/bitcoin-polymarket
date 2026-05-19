"""
Ring Buffer — Pre-allocated numpy circular buffer for zero-copy data retention.

This is the PRIMARY data source for the feature engine. All feature computation
reads from these in-memory arrays — NEVER from TimescaleDB during inference.

Architecture Rules:
  - I/O layer (WebSocket handlers) writes to this buffer via push()
  - CPU layer (feature engine) reads from this buffer via get_window()
  - DB writes are async fire-and-forget — separate from this buffer
  - Lock is only held during writes; reads on contiguous numpy slices are safe

Capacity: 600 bars (50 hours at 5-min candles, or 10 hours at 1-min candles).
This covers the maximum rolling window requirement of 200 bars with 3× headroom.
"""

import numpy as np
from threading import RLock
from typing import Optional
import time
import structlog

logger = structlog.get_logger(__name__)


class RingBuffer:
    """
    Lock-free-read circular buffer backed by pre-allocated numpy arrays.

    Holds last N bars of OHLCV + order book + trade flow + derivatives + 
    cross-asset data. All arrays are pre-allocated at init — zero runtime 
    memory allocation on the hot path.

    Thread Safety:
      - push() acquires RLock (write path, called from I/O thread)
      - get_window() acquires RLock briefly to read head/size, then
        operates on numpy slices (effectively zero-copy)
    """

    CAPACITY = 600  # 600 bars — covers all rolling window requirements

    # Field definitions: (name, dtype)
    # Grouped by data source for clarity
    _FIELDS = {
        # === OHLCV (Binance spot kline) ===
        'timestamp':  np.int64,
        'open':       np.float64,
        'high':       np.float64,
        'low':        np.float64,
        'close':      np.float64,
        'volume':     np.float64,
        'taker_buy':  np.float64,
        'trade_count': np.float64,

        # === Order Book snapshots (L2) ===
        'bid_depth':  np.float64,
        'ask_depth':  np.float64,
        'spread':     np.float64,
        'ofi':        np.float64,
        'mid_price':  np.float64,
        'bid_depth_1pct': np.float64,
        'ask_depth_1pct': np.float64,
        'bid_ask_ratio_5': np.float64,
        'bid_ask_ratio_10': np.float64,

        # === Derivatives (Binance Futures + Deribit) ===
        'funding':    np.float64,
        'oi':         np.float64,
        'liq_long':   np.float64,
        'liq_short':  np.float64,
        'iv_25delta': np.float64,
        'put_call_ratio': np.float64,

        # === Cross-asset (ETH, SOL) ===
        'eth_close':  np.float64,
        'eth_volume': np.float64,
        'sol_close':  np.float64,
        'sol_volume': np.float64,

        # === Microstructure (aggregated trades) ===
        'large_trade_count': np.float64,
        'avg_trade_size':    np.float64,
        'taker_buy_ratio':   np.float64,
    }

    def __init__(self, capacity: Optional[int] = None):
        """
        Initialize ring buffer with pre-allocated numpy arrays.

        Args:
            capacity: Override default capacity (for testing).
        """
        if capacity is not None:
            self.CAPACITY = capacity

        # Pre-allocate all arrays at init — no runtime allocation
        self._arrays: dict[str, np.ndarray] = {}
        for field_name, dtype in self._FIELDS.items():
            self._arrays[field_name] = np.zeros(self.CAPACITY, dtype=dtype)

        self._head: int = 0      # Next write position (circular)
        self._size: int = 0      # Number of valid entries (up to CAPACITY)
        self._lock = RLock()

        # Performance tracking
        self._total_pushes: int = 0
        self._last_push_ns: int = 0

        logger.info(
            "ring_buffer_initialized",
            capacity=self.CAPACITY,
            fields=len(self._FIELDS),
            memory_mb=round(self._memory_usage_bytes() / 1024 / 1024, 2),
        )

    def push(self, bar: dict) -> None:
        """
        Write a single bar to the ring buffer.

        This is called from the I/O thread (WebSocket handler).
        Only the fields present in `bar` are updated — missing fields
        retain their previous (zero) value, which the feature engine
        handles gracefully (last-known-value fallback).

        Args:
            bar: Dictionary with field names matching _FIELDS keys.
                 At minimum, 'timestamp' and 'close' should be present.
        """
        t0 = time.perf_counter_ns()

        with self._lock:
            idx = self._head

            # Write all provided fields in-place — no allocation
            for field_name, value in bar.items():
                if field_name in self._arrays:
                    self._arrays[field_name][idx] = value

            # Advance head pointer (circular)
            self._head = (self._head + 1) % self.CAPACITY
            self._size = min(self._size + 1, self.CAPACITY)

        self._total_pushes += 1
        self._last_push_ns = time.perf_counter_ns() - t0

    def push_batch(self, bars: list[dict]) -> None:
        """
        Write multiple bars to the ring buffer in a single lock acquisition.

        Used during REST backfill to populate the buffer efficiently.
        Bars must be in chronological order (oldest first).

        Args:
            bars: List of bar dictionaries, ordered oldest-to-newest.
        """
        t0 = time.perf_counter_ns()

        with self._lock:
            for bar in bars:
                idx = self._head
                for field_name, value in bar.items():
                    if field_name in self._arrays:
                        self._arrays[field_name][idx] = value
                self._head = (self._head + 1) % self.CAPACITY
                self._size = min(self._size + 1, self.CAPACITY)

        self._total_pushes += len(bars)
        elapsed_ns = time.perf_counter_ns() - t0

        logger.info(
            "ring_buffer_batch_push",
            bars_count=len(bars),
            elapsed_ms=round(elapsed_ns / 1_000_000, 2),
            buffer_size=self._size,
        )

    def get_window(self, n: int) -> dict[str, np.ndarray]:
        """
        Return last n bars as contiguous numpy arrays.

        This is the PRIMARY data access method for the feature engine.
        Returns a dict of field_name -> numpy array, each of length n.

        For wrap-around cases, arrays are concatenated to ensure contiguity.
        This is the only allocation in the read path, and it happens only
        when the window spans the buffer boundary.

        Args:
            n: Number of bars to retrieve (capped at current size).

        Returns:
            Dictionary mapping field names to numpy arrays of length n.
        """
        with self._lock:
            n = min(n, self._size)
            if n == 0:
                return {name: np.array([], dtype=dtype)
                        for name, dtype in self._FIELDS.items()}

            end = self._head
            start = (end - n) % self.CAPACITY

        # Build result outside the lock — numpy slicing is fast
        result = {}
        if start < end:
            # No wrap-around — direct slice (zero-copy view)
            for name, arr in self._arrays.items():
                result[name] = arr[start:end].copy()
        else:
            # Wrap-around — concatenate two slices
            for name, arr in self._arrays.items():
                result[name] = np.concatenate([arr[start:], arr[:end]])

        return result

    def get_latest(self) -> dict[str, float]:
        """
        Return the most recent bar as a dictionary of scalars.

        Used for quick state checks and Redis caching.
        """
        with self._lock:
            if self._size == 0:
                return {}
            idx = (self._head - 1) % self.CAPACITY

        return {name: float(arr[idx]) for name, arr in self._arrays.items()}

    def get_field_window(self, field: str, n: int) -> np.ndarray:
        """
        Return last n values of a single field as a contiguous numpy array.

        More efficient than get_window() when only one field is needed.

        Args:
            field: Field name (must exist in _FIELDS).
            n: Number of values to retrieve.

        Returns:
            Numpy array of length min(n, current_size).

        Raises:
            KeyError: If field name is not recognized.
        """
        if field not in self._arrays:
            raise KeyError(f"Unknown field: {field}. Valid fields: {list(self._FIELDS.keys())}")

        with self._lock:
            n = min(n, self._size)
            if n == 0:
                return np.array([], dtype=self._FIELDS[field])

            end = self._head
            start = (end - n) % self.CAPACITY

        arr = self._arrays[field]
        if start < end:
            return arr[start:end].copy()
        else:
            return np.concatenate([arr[start:], arr[:end]])

    def update_latest(self, updates: dict) -> None:
        """
        Update fields of the most recent bar in-place.

        Used when supplementary data (order book, derivatives) arrives
        after the initial OHLCV bar has been written. Only updates 
        the specified fields — other fields remain unchanged.

        Args:
            updates: Dictionary of field_name -> new_value.
        """
        with self._lock:
            if self._size == 0:
                return
            idx = (self._head - 1) % self.CAPACITY
            for field_name, value in updates.items():
                if field_name in self._arrays:
                    self._arrays[field_name][idx] = value

    @property
    def size(self) -> int:
        """Current number of valid entries in the buffer."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return self._size >= self.CAPACITY

    @property
    def is_empty(self) -> bool:
        """Whether the buffer contains no data."""
        return self._size == 0

    @property
    def latest_timestamp(self) -> Optional[int]:
        """Timestamp of the most recent bar, or None if empty."""
        if self._size == 0:
            return None
        idx = (self._head - 1) % self.CAPACITY
        return int(self._arrays['timestamp'][idx])

    @property
    def oldest_timestamp(self) -> Optional[int]:
        """Timestamp of the oldest bar, or None if empty."""
        if self._size == 0:
            return None
        if self._size < self.CAPACITY:
            return int(self._arrays['timestamp'][0])
        return int(self._arrays['timestamp'][self._head])

    def _memory_usage_bytes(self) -> int:
        """Calculate total memory usage of all pre-allocated arrays."""
        return sum(arr.nbytes for arr in self._arrays.values())

    def stats(self) -> dict:
        """Return buffer statistics for monitoring."""
        return {
            'capacity': self.CAPACITY,
            'size': self._size,
            'is_full': self.is_full,
            'total_pushes': self._total_pushes,
            'last_push_latency_us': round(self._last_push_ns / 1000, 2),
            'memory_mb': round(self._memory_usage_bytes() / 1024 / 1024, 2),
            'latest_timestamp': self.latest_timestamp,
            'oldest_timestamp': self.oldest_timestamp,
            'fields': list(self._FIELDS.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.CAPACITY}, size={self._size}, "
            f"fields={len(self._FIELDS)}, "
            f"memory={self._memory_usage_bytes() / 1024:.1f}KB)"
        )
