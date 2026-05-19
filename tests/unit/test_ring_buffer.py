"""
Unit Tests — Ring Buffer

Tests the core Ring Buffer implementation for:
  - Basic push/get operations
  - Wrap-around behavior
  - Batch push
  - Field-level access
  - Update latest
  - Thread safety
  - Performance (latency < 5ms)
"""

import time
import threading
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.ring_buffer import RingBuffer


class TestRingBufferBasic:
    """Basic ring buffer operations."""

    def test_init_empty(self):
        """Buffer starts empty."""
        rb = RingBuffer()
        assert rb.size == 0
        assert rb.is_empty
        assert not rb.is_full
        assert rb.latest_timestamp is None
        assert rb.oldest_timestamp is None

    def test_init_custom_capacity(self):
        """Custom capacity is respected."""
        rb = RingBuffer(capacity=100)
        assert rb.CAPACITY == 100

    def test_push_single(self):
        """Single push increases size."""
        rb = RingBuffer(capacity=10)
        rb.push({'timestamp': 1000, 'close': 100.0, 'volume': 50.0})
        assert rb.size == 1
        assert rb.latest_timestamp == 1000

    def test_push_multiple(self):
        """Multiple pushes accumulate."""
        rb = RingBuffer(capacity=10)
        for i in range(5):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})
        assert rb.size == 5
        assert rb.latest_timestamp == 4000

    def test_push_fills_capacity(self):
        """Buffer fills to capacity."""
        rb = RingBuffer(capacity=5)
        for i in range(5):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})
        assert rb.size == 5
        assert rb.is_full

    def test_push_wraps_around(self):
        """Buffer wraps around when full."""
        rb = RingBuffer(capacity=5)
        for i in range(8):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})
        assert rb.size == 5  # Still at capacity
        assert rb.latest_timestamp == 7000
        # Oldest should be the 4th item (index 3)
        assert rb.oldest_timestamp == 3000

    def test_get_window_basic(self):
        """get_window returns correct number of bars."""
        rb = RingBuffer(capacity=10)
        for i in range(5):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})

        w = rb.get_window(3)
        assert len(w['close']) == 3
        assert len(w['timestamp']) == 3

    def test_get_window_values(self):
        """get_window returns correct values in order."""
        rb = RingBuffer(capacity=10)
        for i in range(5):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})

        w = rb.get_window(5)
        np.testing.assert_array_almost_equal(
            w['close'], [100.0, 101.0, 102.0, 103.0, 104.0]
        )

    def test_get_window_capped_at_size(self):
        """get_window caps at current size."""
        rb = RingBuffer(capacity=10)
        for i in range(3):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})

        w = rb.get_window(10)  # Ask for more than available
        assert len(w['close']) == 3

    def test_get_window_empty(self):
        """get_window returns empty arrays when buffer is empty."""
        rb = RingBuffer(capacity=10)
        w = rb.get_window(5)
        assert len(w['close']) == 0

    def test_get_window_wrap_around(self):
        """get_window returns correct values after wrap-around."""
        rb = RingBuffer(capacity=5)
        for i in range(8):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})

        w = rb.get_window(5)
        # Should have items 3, 4, 5, 6, 7
        np.testing.assert_array_almost_equal(
            w['close'], [103.0, 104.0, 105.0, 106.0, 107.0]
        )

    def test_get_latest(self):
        """get_latest returns the most recent bar."""
        rb = RingBuffer(capacity=10)
        rb.push({'timestamp': 1000, 'close': 100.0})
        rb.push({'timestamp': 2000, 'close': 200.0})

        latest = rb.get_latest()
        assert latest['close'] == 200.0
        assert latest['timestamp'] == 2000.0

    def test_get_latest_empty(self):
        """get_latest returns empty dict when buffer is empty."""
        rb = RingBuffer(capacity=10)
        assert rb.get_latest() == {}


class TestRingBufferBatch:
    """Batch push operations."""

    def test_push_batch(self):
        """Batch push loads multiple bars efficiently."""
        rb = RingBuffer(capacity=100)
        bars = [
            {'timestamp': i * 1000, 'close': 100.0 + i, 'volume': float(i)}
            for i in range(50)
        ]
        rb.push_batch(bars)
        assert rb.size == 50

        w = rb.get_window(50)
        assert len(w['close']) == 50
        np.testing.assert_array_almost_equal(
            w['close'][:3], [100.0, 101.0, 102.0]
        )

    def test_push_batch_exceeds_capacity(self):
        """Batch push with more bars than capacity keeps only the latest."""
        rb = RingBuffer(capacity=10)
        bars = [
            {'timestamp': i * 1000, 'close': 100.0 + i}
            for i in range(20)
        ]
        rb.push_batch(bars)
        assert rb.size == 10

        w = rb.get_window(10)
        np.testing.assert_array_almost_equal(
            w['close'], [110.0, 111.0, 112.0, 113.0, 114.0,
                         115.0, 116.0, 117.0, 118.0, 119.0]
        )


class TestRingBufferFieldAccess:
    """Per-field access and updates."""

    def test_get_field_window(self):
        """get_field_window returns a single field."""
        rb = RingBuffer(capacity=10)
        for i in range(5):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i})

        close = rb.get_field_window('close', 3)
        assert len(close) == 3
        np.testing.assert_array_almost_equal(close, [102.0, 103.0, 104.0])

    def test_get_field_window_invalid(self):
        """get_field_window raises KeyError for unknown field."""
        rb = RingBuffer(capacity=10)
        with pytest.raises(KeyError):
            rb.get_field_window('nonexistent', 5)

    def test_update_latest(self):
        """update_latest modifies the most recent bar."""
        rb = RingBuffer(capacity=10)
        rb.push({'timestamp': 1000, 'close': 100.0})

        rb.update_latest({'bid_depth': 5.0, 'ask_depth': 3.0})

        latest = rb.get_latest()
        assert latest['bid_depth'] == 5.0
        assert latest['ask_depth'] == 3.0
        assert latest['close'] == 100.0  # Unchanged

    def test_update_latest_empty(self):
        """update_latest on empty buffer is a no-op."""
        rb = RingBuffer(capacity=10)
        rb.update_latest({'close': 100.0})  # Should not crash
        assert rb.is_empty

    def test_partial_fields(self):
        """push with missing fields defaults to zero."""
        rb = RingBuffer(capacity=10)
        rb.push({'timestamp': 1000, 'close': 100.0})

        latest = rb.get_latest()
        assert latest['close'] == 100.0
        assert latest['volume'] == 0.0  # Default zero
        assert latest['bid_depth'] == 0.0  # Default zero


class TestRingBufferThreadSafety:
    """Thread safety tests."""

    def test_concurrent_push_and_read(self):
        """Concurrent push and get_window should not crash."""
        rb = RingBuffer(capacity=100)
        errors = []

        def writer():
            try:
                for i in range(1000):
                    rb.push({'timestamp': i * 1000, 'close': 100.0 + i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(500):
                    w = rb.get_window(50)
                    # Just verify it returns valid arrays
                    assert isinstance(w['close'], np.ndarray)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestRingBufferPerformance:
    """Performance benchmarks."""

    def test_push_latency(self):
        """Single push should complete in < 5ms (< 50μs typical)."""
        rb = RingBuffer()
        bar = {
            'timestamp': 1000,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 50.0,
            'taker_buy': 25.0,
        }

        times = []
        for i in range(1000):
            bar['timestamp'] = i * 1000
            t0 = time.perf_counter()
            rb.push(bar)
            times.append(time.perf_counter() - t0)

        p95_ms = np.percentile(times, 95) * 1000
        assert p95_ms < 5.0, f"P95 push latency {p95_ms:.3f}ms exceeds 5ms budget"

    def test_get_window_latency(self):
        """get_window(200) should complete in < 5ms."""
        rb = RingBuffer()
        for i in range(600):
            rb.push({'timestamp': i * 1000, 'close': 100.0 + i, 'volume': float(i)})

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            w = rb.get_window(200)
            times.append(time.perf_counter() - t0)

        p95_ms = np.percentile(times, 95) * 1000
        assert p95_ms < 5.0, f"P95 get_window latency {p95_ms:.3f}ms exceeds 5ms budget"

    def test_batch_push_latency(self):
        """Batch push of 600 bars should complete in < 50ms."""
        rb = RingBuffer()
        bars = [
            {'timestamp': i * 1000, 'close': 100.0 + i, 'volume': float(i)}
            for i in range(600)
        ]

        t0 = time.perf_counter()
        rb.push_batch(bars)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 50.0, f"Batch push latency {elapsed_ms:.2f}ms exceeds 50ms budget"


class TestRingBufferStats:
    """Stats and repr methods."""

    def test_stats(self):
        """stats() returns expected structure."""
        rb = RingBuffer(capacity=10)
        rb.push({'timestamp': 1000, 'close': 100.0})

        stats = rb.stats()
        assert stats['capacity'] == 10
        assert stats['size'] == 1
        assert stats['total_pushes'] == 1
        assert 'fields' in stats
        assert 'memory_mb' in stats

    def test_repr(self):
        """repr gives a useful string."""
        rb = RingBuffer(capacity=10)
        r = repr(rb)
        assert 'capacity=10' in r
        assert 'size=0' in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
