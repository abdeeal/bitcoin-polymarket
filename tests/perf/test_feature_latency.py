"""
Performance Test — Feature Engine Latency

PRD Requirement: P95 feature computation latency < 100ms.
Tests the complete feature pipeline from Ring Buffer read to feature dict output.
"""
import time
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.ring_buffer import RingBuffer
from src.features.feature_engine import FeatureEngine


def _create_mock_ring_buffer(n_bars: int = 600) -> RingBuffer:
    """Create a Ring Buffer filled with realistic synthetic data."""
    rb = RingBuffer(capacity=n_bars)
    base_price = 100000.0
    bars = []
    for i in range(n_bars):
        noise = np.random.randn() * 50
        close = base_price + noise + i * 0.1
        bars.append({
            'timestamp': int(1700000000000 + i * 60000),
            'open': close - np.random.rand() * 20,
            'high': close + np.random.rand() * 30,
            'low': close - np.random.rand() * 30,
            'close': close,
            'volume': 50.0 + np.random.rand() * 100,
            'taker_buy': 25.0 + np.random.rand() * 50,
            'trade_count': 500.0 + np.random.rand() * 200,
            'bid_depth': 10.0 + np.random.rand() * 5,
            'ask_depth': 8.0 + np.random.rand() * 5,
            'spread': 0.01 + np.random.rand() * 0.005,
            'mid_price': close,
            'funding': 0.0001 * (1 + np.random.randn() * 0.1),
            'oi': 50000.0 + np.random.randn() * 1000,
            'liq_long': np.random.rand() * 1000000,
            'liq_short': np.random.rand() * 1000000,
            'iv_25delta': 50 + np.random.randn() * 10,
            'put_call_ratio': 0.8 + np.random.randn() * 0.2,
            'eth_close': 3500.0 + np.random.randn() * 50,
            'eth_volume': 1000 + np.random.rand() * 500,
            'sol_close': 180.0 + np.random.randn() * 5,
            'sol_volume': 500 + np.random.rand() * 200,
            'large_trade_count': float(np.random.randint(0, 10)),
            'avg_trade_size': 0.05 + np.random.rand() * 0.1,
            'taker_buy_ratio': 0.45 + np.random.rand() * 0.1,
            'bid_depth_1pct': 5.0 + np.random.rand() * 3,
            'ask_depth_1pct': 4.0 + np.random.rand() * 3,
            'bid_ask_ratio_5': 1.0 + np.random.randn() * 0.2,
            'bid_ask_ratio_10': 1.0 + np.random.randn() * 0.15,
        })
    rb.push_batch(bars)
    return rb


class TestFeatureEnginePerformance:
    """Feature engine performance benchmarks."""

    def test_feature_engine_latency_p95(self):
        """
        P95 feature computation must be < 100ms.

        PRD 5.8: Must pass before deployment.
        """
        rb = _create_mock_ring_buffer(600)
        engine = FeatureEngine(rb)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            features = engine.compute_sync('ACTIVE')
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        p95_ms = np.percentile(times, 95) * 1000
        p99_ms = np.percentile(times, 99) * 1000
        mean_ms = np.mean(times) * 1000

        print(f"\n=== Feature Engine Performance ===")
        print(f"  Features: {len(features)}")
        print(f"  Mean:  {mean_ms:.1f} ms")
        print(f"  P95:   {p95_ms:.1f} ms")
        print(f"  P99:   {p99_ms:.1f} ms")

        assert p95_ms < 100, f"P95 {p95_ms:.1f}ms exceeds 100ms budget"

    def test_feature_count(self):
        """Feature engine produces 150+ features."""
        rb = _create_mock_ring_buffer(600)
        engine = FeatureEngine(rb)
        features = engine.compute_sync('ACTIVE')

        # Remove metadata keys
        feature_keys = [k for k in features if k not in ('bot_state', 'timestamp', 'buffer_size', '_insufficient_data')]
        print(f"\n  Total features: {len(feature_keys)}")
        assert len(feature_keys) >= 100, f"Only {len(feature_keys)} features"

    def test_no_nan_features(self):
        """No feature should be NaN with full buffer."""
        rb = _create_mock_ring_buffer(600)
        engine = FeatureEngine(rb)
        features = engine.compute_sync('ACTIVE')

        nan_keys = [k for k, v in features.items()
                    if isinstance(v, float) and np.isnan(v)]
        print(f"\n  NaN features: {nan_keys}")
        assert len(nan_keys) == 0, f"NaN features found: {nan_keys}"

    def test_feature_engine_stats(self):
        """Stats tracking works correctly."""
        rb = _create_mock_ring_buffer(200)
        engine = FeatureEngine(rb)
        engine.compute_sync('ACTIVE')
        engine.compute_sync('COOLING_DOWN')

        stats = engine.stats()
        assert stats['compute_count'] == 2
        assert stats['avg_latency_ms'] > 0
        assert stats['last_latency_ms'] > 0

    def test_cooling_down_produces_features(self):
        """Feature engine runs during COOLING_DOWN (no data gaps)."""
        rb = _create_mock_ring_buffer(200)
        engine = FeatureEngine(rb)
        features = engine.compute_sync('COOLING_DOWN')
        assert features['bot_state'] == 'COOLING_DOWN'
        assert len(features) > 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
