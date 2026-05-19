"""
Performance Tests — Ingestion Pipeline

Tests the ingestion pipeline performance:
  - Ring Buffer write latency < 5ms (P95)
  - Ring Buffer read latency < 5ms (P95)
  - Batch backfill < 50ms for 600 candles
  - Full pipeline round-trip latency
"""

import time
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.ring_buffer import RingBuffer


class TestIngestionPipelinePerformance:
    """Performance tests for the ingestion pipeline."""

    def _generate_bar(self, i: int) -> dict:
        """Generate a realistic test bar."""
        return {
            'timestamp': i * 60_000,
            'open': 100000.0 + i * 10,
            'high': 100050.0 + i * 10,
            'low': 99950.0 + i * 10,
            'close': 100025.0 + i * 10,
            'volume': 50.0 + (i % 20),
            'taker_buy': 25.0 + (i % 10),
            'trade_count': 500.0 + (i % 100),
            'bid_depth': 10.0 + (i % 5),
            'ask_depth': 8.0 + (i % 5),
            'spread': 0.01,
            'mid_price': 100025.0 + i * 10,
            'funding': 0.0001,
            'oi': 50000.0,
            'eth_close': 3500.0 + i,
            'sol_close': 180.0 + i * 0.1,
        }

    def test_ring_buffer_write_latency(self):
        """
        Ring Buffer single push should complete in < 5ms P95.

        PRD requirement: ingestion pipeline adds <5ms latency to Ring Buffer writes.
        """
        rb = RingBuffer()
        times = []

        for i in range(5000):
            bar = self._generate_bar(i)
            t0 = time.perf_counter()
            rb.push(bar)
            times.append(time.perf_counter() - t0)

        p95_ms = np.percentile(times, 95) * 1000
        p99_ms = np.percentile(times, 99) * 1000
        mean_us = np.mean(times) * 1_000_000

        print(f"\n=== Ring Buffer Write Performance ===")
        print(f"  Mean:  {mean_us:.1f} us")
        print(f"  P95:   {p95_ms:.3f} ms")
        print(f"  P99:   {p99_ms:.3f} ms")

        assert p95_ms < 5.0, f"P95 write latency {p95_ms:.3f}ms exceeds 5ms budget"

    def test_ring_buffer_read_latency(self):
        """
        Ring Buffer get_window(200) should complete in < 5ms P95.

        This is the primary read path for the feature engine.
        """
        rb = RingBuffer()

        # Fill buffer completely
        for i in range(600):
            rb.push(self._generate_bar(i))

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            w = rb.get_window(200)
            times.append(time.perf_counter() - t0)

        p95_ms = np.percentile(times, 95) * 1000
        p99_ms = np.percentile(times, 99) * 1000
        mean_us = np.mean(times) * 1_000_000

        print(f"\n=== Ring Buffer Read Performance ===")
        print(f"  Mean:  {mean_us:.1f} us")
        print(f"  P95:   {p95_ms:.3f} ms")
        print(f"  P99:   {p99_ms:.3f} ms")

        assert p95_ms < 5.0, f"P95 read latency {p95_ms:.3f}ms exceeds 5ms budget"

    def test_batch_push_600_candles(self):
        """
        Batch push of 600 candles should complete in < 50ms.

        This simulates the cold-boot backfill writing to Ring Buffer.
        """
        rb = RingBuffer()
        bars = [self._generate_bar(i) for i in range(600)]

        t0 = time.perf_counter()
        rb.push_batch(bars)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n=== Batch Push 600 Candles ===")
        print(f"  Elapsed: {elapsed_ms:.2f} ms")
        print(f"  Per bar: {elapsed_ms / 600:.3f} ms")

        assert elapsed_ms < 50.0, f"Batch push took {elapsed_ms:.2f}ms, exceeds 50ms budget"

    def test_full_round_trip(self):
        """
        Full round-trip: push → get_window → verify data integrity.

        Ensures no data loss or corruption in the push → read path.
        """
        rb = RingBuffer()

        # Push 1000 bars (wraps around)
        for i in range(1000):
            rb.push({
                'timestamp': i * 60_000,
                'close': 100000.0 + i,
                'volume': float(i),
            })

        # Read last 200
        w = rb.get_window(200)

        # Verify data integrity
        expected_close_first = 100000.0 + 800  # 1000 - 200
        expected_close_last = 100000.0 + 999

        assert len(w['close']) == 200
        assert abs(w['close'][0] - expected_close_first) < 0.001
        assert abs(w['close'][-1] - expected_close_last) < 0.001

        # Verify timestamps are monotonically increasing
        timestamps = w['timestamp']
        assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))

    def test_memory_usage(self):
        """
        Ring Buffer memory usage should be reasonable (< 1 MB).
        """
        rb = RingBuffer()
        stats = rb.stats()
        memory_mb = stats['memory_mb']

        print(f"\n=== Ring Buffer Memory ===")
        print(f"  Total: {memory_mb:.2f} MB")
        print(f"  Fields: {len(stats['fields'])}")
        print(f"  Capacity: {stats['capacity']}")

        assert memory_mb < 1.0, f"Memory usage {memory_mb:.2f}MB exceeds 1MB budget"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
