"""
Unit Tests — Persistence Worker

Tests the async persistence worker:
  - Batch accumulation and flushing
  - Timeout-based flushing
  - Graceful shutdown
  - Error handling
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.db.persistence_worker import PersistenceWorker


class MockDatabaseClient:
    """Mock database client for testing."""

    def __init__(self, should_fail: bool = False):
        self.inserted_batches: list = []
        self.should_fail = should_fail

    async def bulk_insert(self, items):
        if self.should_fail:
            raise Exception("DB write failed")
        self.inserted_batches.append(items)


class TestPersistenceWorkerBasic:
    """Basic persistence worker tests."""

    @pytest.mark.asyncio
    async def test_single_item_flush(self):
        """Single item is flushed after timeout."""
        db = MockDatabaseClient()
        queue = asyncio.Queue(maxsize=100)
        worker = PersistenceWorker(db, queue, flush_timeout=0.1)

        # Start worker
        task = asyncio.create_task(worker.run())

        # Push an item
        await queue.put(('candle', {'timestamp': 1000, 'close': 100.0}))

        # Wait for flush
        await asyncio.sleep(0.3)

        # Stop worker
        await worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(db.inserted_batches) >= 1
        total_items = sum(len(b) for b in db.inserted_batches)
        assert total_items == 1

    @pytest.mark.asyncio
    async def test_batch_accumulation(self):
        """Multiple items are batched together."""
        db = MockDatabaseClient()
        queue = asyncio.Queue(maxsize=100)
        worker = PersistenceWorker(db, queue, max_batch_size=5, flush_timeout=0.1)

        task = asyncio.create_task(worker.run())

        # Push 5 items rapidly
        for i in range(5):
            await queue.put(('candle', {'timestamp': i * 1000, 'close': 100.0 + i}))

        await asyncio.sleep(0.3)
        await worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        total_items = sum(len(b) for b in db.inserted_batches)
        assert total_items == 5

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Worker continues after DB errors."""
        db = MockDatabaseClient(should_fail=True)
        queue = asyncio.Queue(maxsize=100)
        worker = PersistenceWorker(db, queue, flush_timeout=0.1)

        task = asyncio.create_task(worker.run())

        await queue.put(('candle', {'timestamp': 1000, 'close': 100.0}))
        await asyncio.sleep(0.5)

        # Worker should still be running despite error
        assert worker._running

        await worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert worker._total_errors > 0

    @pytest.mark.asyncio
    async def test_stats(self):
        """Stats are tracked correctly."""
        db = MockDatabaseClient()
        queue = asyncio.Queue(maxsize=100)
        worker = PersistenceWorker(db, queue, flush_timeout=0.1)

        task = asyncio.create_task(worker.run())

        await queue.put(('candle', {'timestamp': 1000, 'close': 100.0}))
        await asyncio.sleep(0.3)

        stats = worker.stats()
        assert 'running' in stats
        assert 'queue_size' in stats
        assert 'total_items' in stats

        await worker.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
