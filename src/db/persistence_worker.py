"""
Persistence Worker — Background async task for non-blocking DB writes.

Architecture Rules:
  - Runs as a background asyncio task
  - Consumes from an asyncio.Queue — never called directly from hot path
  - Batches items for efficient bulk inserts
  - Non-blocking: does not affect I/O or CPU paths
  - Handles errors gracefully — never crashes the main loop

Flow:
  1. I/O handlers push (type, data) tuples to the queue
  2. This worker drains the queue in batches (up to 50 items)
  3. Batched items are bulk-inserted to TimescaleDB via DatabaseClient
  4. Timeout-based flushing ensures data is persisted even at low volume
"""

import asyncio
import time
from typing import Optional

import structlog

from src.db.client import DatabaseClient

logger = structlog.get_logger(__name__)

# Maximum batch size before flushing to DB
MAX_BATCH_SIZE = 50

# Timeout before flushing a partial batch (seconds)
FLUSH_TIMEOUT = 1.0


class PersistenceWorker:
    """
    Background async worker that drains a write queue and bulk-inserts
    to TimescaleDB.

    This is the ONLY component that writes to the database during live
    operation. All other components push to the queue and forget.

    The worker runs continuously until stopped, batching items for
    efficient database writes.
    """

    def __init__(
        self,
        db_client: DatabaseClient,
        queue: asyncio.Queue,
        max_batch_size: int = MAX_BATCH_SIZE,
        flush_timeout: float = FLUSH_TIMEOUT,
    ):
        """
        Initialize the persistence worker.

        Args:
            db_client: DatabaseClient instance with active connection pool.
            queue: asyncio.Queue shared with I/O handlers.
            max_batch_size: Maximum items to batch before flushing.
            flush_timeout: Seconds to wait before flushing partial batch.
        """
        self._db = db_client
        self._queue = queue
        self._max_batch = max_batch_size
        self._flush_timeout = flush_timeout
        self._running: bool = False

        # Statistics
        self._total_items: int = 0
        self._total_batches: int = 0
        self._total_errors: int = 0
        self._last_flush_time: float = 0

    async def run(self) -> None:
        """
        Main worker loop — runs indefinitely until stopped.

        Strategy:
          1. Wait for at least one item in the queue (with timeout)
          2. Drain queue up to max_batch_size items
          3. Bulk insert the batch
          4. Repeat
        """
        self._running = True
        batch: list[tuple[str, dict]] = []

        logger.info(
            "persistence_worker_started",
            max_batch=self._max_batch,
            flush_timeout=self._flush_timeout,
        )

        while self._running:
            try:
                # Wait for at least one item
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=self._flush_timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    # Timeout — flush whatever we have
                    if batch:
                        await self._flush(batch)
                        batch = []
                    continue

                # Drain queue up to max batch size
                while not self._queue.empty() and len(batch) < self._max_batch:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                # Flush if batch is full or queue is empty
                if len(batch) >= self._max_batch or self._queue.empty():
                    await self._flush(batch)
                    batch = []

            except asyncio.CancelledError:
                # Graceful shutdown — flush remaining
                if batch:
                    await self._flush(batch)
                logger.info("persistence_worker_cancelled")
                break
            except Exception as e:
                self._total_errors += 1
                logger.error(
                    "persistence_worker_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    batch_size=len(batch),
                )
                # Don't lose data — keep batch for next attempt
                await asyncio.sleep(1)

        logger.info(
            "persistence_worker_stopped",
            total_items=self._total_items,
            total_batches=self._total_batches,
            total_errors=self._total_errors,
        )

    async def _flush(self, batch: list[tuple[str, dict]]) -> None:
        """
        Flush a batch of items to the database.

        Handles errors gracefully — logs and moves on.
        Items are not retried individually to prevent blocking.
        """
        if not batch:
            return

        t0 = time.perf_counter()

        try:
            await self._db.bulk_insert(batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self._total_items += len(batch)
            self._total_batches += 1
            self._last_flush_time = time.time()

            logger.debug(
                "persistence_flush",
                items=len(batch),
                elapsed_ms=round(elapsed_ms, 2),
                total_items=self._total_items,
                queue_size=self._queue.qsize(),
            )

        except Exception as e:
            self._total_errors += 1
            logger.error(
                "persistence_flush_error",
                items=len(batch),
                error=str(e),
            )

    async def stop(self) -> None:
        """Signal the worker to stop after flushing remaining items."""
        self._running = False

    @property
    def queue_size(self) -> int:
        """Current number of items waiting in the queue."""
        return self._queue.qsize()

    def stats(self) -> dict:
        """Return worker statistics for monitoring."""
        return {
            'running': self._running,
            'queue_size': self._queue.qsize(),
            'total_items': self._total_items,
            'total_batches': self._total_batches,
            'total_errors': self._total_errors,
            'last_flush_time': self._last_flush_time,
        }
