"""
Unit Tests — Data Manager

Tests the DataManager orchestration logic:
  - Trade aggregation
  - Candle enrichment with OB + cross-asset data
  - Data gap detection
  - Backfill data merging
"""

import asyncio
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.ring_buffer import RingBuffer
from src.ingestion.data_manager import DataManager, TradeAggregator


class TestTradeAggregator:
    """Tests for the TradeAggregator component."""

    def test_init_empty(self):
        """Aggregator starts with zero counts."""
        agg = TradeAggregator()
        snapshot = agg.snapshot()
        assert snapshot['trade_count'] == 0
        assert snapshot['large_trade_count'] == 0

    def test_add_trade_counts(self):
        """Trades are counted correctly."""
        agg = TradeAggregator()
        agg.add_trade({'price': 100000, 'quantity': 0.1, 'is_buyer_maker': False})
        agg.add_trade({'price': 100001, 'quantity': 0.2, 'is_buyer_maker': True})
        agg.add_trade({'price': 100002, 'quantity': 0.3, 'is_buyer_maker': False})

        snapshot = agg.snapshot()
        assert snapshot['trade_count'] == 3.0

    def test_large_trade_detection(self):
        """Trades >= 0.5 BTC are counted as large."""
        agg = TradeAggregator()
        agg.add_trade({'price': 100000, 'quantity': 0.1, 'is_buyer_maker': False})
        agg.add_trade({'price': 100001, 'quantity': 0.5, 'is_buyer_maker': False})
        agg.add_trade({'price': 100002, 'quantity': 1.0, 'is_buyer_maker': True})

        snapshot = agg.snapshot()
        assert snapshot['large_trade_count'] == 2.0

    def test_taker_buy_ratio(self):
        """Taker buy ratio reflects buy vs sell volume."""
        agg = TradeAggregator()
        # 0.4 BTC bought (taker buy, is_buyer_maker=False)
        agg.add_trade({'price': 100000, 'quantity': 0.4, 'is_buyer_maker': False})
        # 0.1 BTC sold (taker sell, is_buyer_maker=True)
        agg.add_trade({'price': 100001, 'quantity': 0.1, 'is_buyer_maker': True})

        snapshot = agg.snapshot()
        assert abs(snapshot['taker_buy_ratio'] - 0.8) < 0.01  # 0.4 / 0.5

    def test_snapshot_resets(self):
        """snapshot() resets all accumulators."""
        agg = TradeAggregator()
        agg.add_trade({'price': 100000, 'quantity': 0.1, 'is_buyer_maker': False})
        agg.snapshot()  # Should reset

        snapshot2 = agg.snapshot()
        assert snapshot2['trade_count'] == 0


class TestDataManagerGapDetection:
    """Tests for data gap detection."""

    def test_no_gap(self):
        """No gap detected for consecutive candles."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        dm._check_gap(1000)
        dm._check_gap(61000)  # 1 minute later
        assert dm._gap_count == 0

    def test_single_gap(self):
        """Single missing candle is detected."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        dm._check_gap(1000)
        dm._check_gap(121000)  # 2 minutes later (1 gap)
        assert dm._gap_count == 1

    def test_multiple_gaps(self):
        """Multiple missing candles are detected."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        dm._check_gap(1000)
        dm._check_gap(301000)  # 5 minutes later (4 gaps)
        assert dm._gap_count == 4


class TestDataManagerKlineHandler:
    """Tests for BTC kline processing."""

    @pytest.mark.asyncio
    async def test_kline_pushes_to_ring_buffer(self):
        """BTC kline event is pushed to Ring Buffer."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        # Simulate a candle
        await dm._on_btc_kline({
            'timestamp': 1000,
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 50.0,
            'taker_buy': 25.0,
            'trade_count': 100.0,
        })

        assert rb.size == 1
        latest = rb.get_latest()
        assert latest['close'] == 100.5

    @pytest.mark.asyncio
    async def test_kline_merges_depth_data(self):
        """Kline event merges latest depth snapshot."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        # Set latest depth
        await dm._on_depth({
            'bid_depth': 5.0,
            'ask_depth': 3.0,
            'spread': 0.01,
        })

        # Push a candle
        await dm._on_btc_kline({
            'timestamp': 1000,
            'close': 100.0,
            'volume': 50.0,
        })

        latest = rb.get_latest()
        assert latest['bid_depth'] == 5.0
        assert latest['ask_depth'] == 3.0

    @pytest.mark.asyncio
    async def test_kline_merges_cross_asset(self):
        """Kline event merges latest cross-asset prices."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        # Set latest ETH price
        await dm._on_cross_asset_kline({
            'symbol': 'eth',
            'close': 3500.0,
            'volume': 1000.0,
        })

        # Push a BTC candle
        await dm._on_btc_kline({
            'timestamp': 1000,
            'close': 100000.0,
            'volume': 50.0,
        })

        latest = rb.get_latest()
        assert latest['eth_close'] == 3500.0


class TestDataManagerFreshness:
    """Tests for data freshness checking."""

    @pytest.mark.asyncio
    async def test_empty_buffer_not_fresh(self):
        """Empty buffer reports as not fresh."""
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        is_fresh = await dm.check_data_freshness()
        assert not is_fresh

    @pytest.mark.asyncio
    async def test_recent_data_is_fresh(self):
        """Recent data (< 1 hour) reports as fresh."""
        import time
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        rb.push({'timestamp': int(time.time() * 1000) - 300_000})  # 5 min ago

        is_fresh = await dm.check_data_freshness()
        assert is_fresh

    @pytest.mark.asyncio
    async def test_old_data_not_fresh(self):
        """Old data (> 1 hour) reports as not fresh."""
        import time
        rb = RingBuffer(capacity=10)
        queue = asyncio.Queue(maxsize=100)
        dm = DataManager(rb, queue)

        rb.push({'timestamp': int(time.time() * 1000) - 7_200_000})  # 2 hours ago

        is_fresh = await dm.check_data_freshness()
        assert not is_fresh


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
