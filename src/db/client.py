"""
Database Client — Async connection pool for TimescaleDB.

Architecture Rules:
  - Uses asyncpg for async non-blocking database access
  - Connection pool is shared across the application
  - All database queries are async — never block the event loop
  - This client is used by PersistenceWorker and dashboard queries
  - NEVER called from the CPU thread pool (feature engine)
"""

import asyncio
from typing import Optional, Any
from datetime import datetime, timezone

import asyncpg
import structlog

logger = structlog.get_logger(__name__)


class DatabaseClient:
    """
    Async database client wrapping an asyncpg connection pool.

    Provides typed insert methods for each table and bulk insert
    capability for the persistence worker.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "btc_prediction",
        user: str = "btcbot",
        password: str = "btcbot_secret",
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ):
        """
        Initialize database client configuration.

        The actual connection pool is created lazily via connect().
        """
        self._dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self._min_pool = min_pool_size
        self._max_pool = max_pool_size
        self._pool: Optional[asyncpg.Pool] = None

        # Statistics
        self._insert_count: int = 0
        self._error_count: int = 0

    async def connect(self) -> None:
        """Create the connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_pool,
                max_size=self._max_pool,
                command_timeout=30,
            )
            logger.info(
                "database_connected",
                dsn=self._dsn.replace(self._dsn.split('@')[0].split('://')[1], '***'),
                pool_size=f"{self._min_pool}-{self._max_pool}",
            )
        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("database_disconnected")

    @property
    def is_connected(self) -> bool:
        """Whether the pool is active."""
        return self._pool is not None

    # =========================================================================
    # Bulk Insert (used by PersistenceWorker)
    # =========================================================================

    async def bulk_insert(self, items: list[tuple[str, dict]]) -> None:
        """
        Insert a batch of items into their respective tables.

        Each item is a tuple of (table_type, data_dict).
        Items are grouped by type and inserted in batch for efficiency.

        Args:
            items: List of (type_str, data_dict) tuples.
                   type_str: 'candle', 'orderbook', 'features', 'bet', etc.
        """
        if not self._pool:
            logger.warning("bulk_insert_no_pool")
            return

        # Group items by type
        grouped: dict[str, list[dict]] = {}
        for item_type, data in items:
            grouped.setdefault(item_type, []).append(data)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for item_type, batch in grouped.items():
                    try:
                        if item_type == 'candle':
                            await self._insert_candles(conn, batch)
                        elif item_type == 'orderbook':
                            await self._insert_orderbook(conn, batch)
                        elif item_type == 'features':
                            await self._insert_features(conn, batch)
                        elif item_type == 'bet':
                            await self._insert_bets(conn, batch)
                        elif item_type == 'funding':
                            await self._insert_funding(conn, batch)
                        elif item_type == 'oi':
                            await self._insert_oi(conn, batch)
                        elif item_type == 'liquidation':
                            await self._insert_liquidations(conn, batch)
                        elif item_type == 'deribit':
                            pass  # Deribit data merged with funding/OI
                        else:
                            logger.warning("unknown_insert_type", item_type=item_type)

                        self._insert_count += len(batch)

                    except Exception as e:
                        self._error_count += 1
                        logger.error(
                            "bulk_insert_error",
                            item_type=item_type,
                            batch_size=len(batch),
                            error=str(e),
                        )

    # =========================================================================
    # Table-specific Insert Methods
    # =========================================================================

    async def _insert_candles(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert candle records into raw_candles_1m."""
        await conn.executemany(
            """
            INSERT INTO raw_candles_1m (timestamp, open, high, low, close, volume, taker_buy_volume, trade_count)
            VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (timestamp) DO NOTHING
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('open', 0),
                    d.get('high', 0),
                    d.get('low', 0),
                    d.get('close', 0),
                    d.get('volume', 0),
                    d.get('taker_buy', 0),
                    int(d.get('trade_count', 0)),
                )
                for d in batch
            ],
        )

    async def _insert_orderbook(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert order book snapshots."""
        await conn.executemany(
            """
            INSERT INTO raw_orderbook_snapshots 
                (timestamp, bid_depth, ask_depth, spread, mid_price, 
                 bid_depth_1pct, ask_depth_1pct, bid_ask_ratio_5, bid_ask_ratio_10)
            VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (timestamp) DO UPDATE SET
                bid_depth = EXCLUDED.bid_depth,
                ask_depth = EXCLUDED.ask_depth,
                spread = EXCLUDED.spread
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('bid_depth', 0),
                    d.get('ask_depth', 0),
                    d.get('spread', 0),
                    d.get('mid_price', 0),
                    d.get('bid_depth_1pct', 0),
                    d.get('ask_depth_1pct', 0),
                    d.get('bid_ask_ratio_5', 0),
                    d.get('bid_ask_ratio_10', 0),
                )
                for d in batch
            ],
        )

    async def _insert_features(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert computed feature vectors."""
        import json
        await conn.executemany(
            """
            INSERT INTO features_5m (timestamp, bot_state, features)
            VALUES (to_timestamp($1 / 1000.0), $2, $3::jsonb)
            ON CONFLICT (timestamp) DO UPDATE SET
                features = EXCLUDED.features,
                bot_state = EXCLUDED.bot_state
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('bot_state', 'ACTIVE'),
                    json.dumps({k: v for k, v in d.items()
                                if k not in ('timestamp', 'bot_state')}),
                )
                for d in batch
            ],
        )

    async def _insert_bets(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert bet log entries."""
        import json
        await conn.executemany(
            """
            INSERT INTO bet_log 
                (timestamp, market_id, signal, confidence_tier, model_prob,
                 bet_size, odds, outcome, pnl, balance_before, balance_after,
                 bracket_index, paper_mode, feature_snapshot)
            VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('market_id', ''),
                    d.get('signal', ''),
                    d.get('confidence_tier', 'NORMAL'),
                    d.get('model_prob', 0),
                    d.get('bet_size', 0),
                    d.get('odds', 0),
                    d.get('outcome'),
                    d.get('pnl', 0),
                    d.get('balance_before', 0),
                    d.get('balance_after', 0),
                    d.get('bracket_index', 0),
                    d.get('paper_mode', True),
                    json.dumps(d.get('features', {})),
                )
                for d in batch
            ],
        )

    async def _insert_funding(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert funding rate records."""
        await conn.executemany(
            """
            INSERT INTO raw_funding_oi (timestamp, funding_rate, mark_price, index_price)
            VALUES (to_timestamp($1 / 1000.0), $2, $3, $4)
            ON CONFLICT (timestamp) DO UPDATE SET
                funding_rate = EXCLUDED.funding_rate
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('funding', 0),
                    d.get('mark_price', 0),
                    d.get('index_price', 0),
                )
                for d in batch
            ],
        )

    async def _insert_oi(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert open interest records."""
        await conn.executemany(
            """
            INSERT INTO raw_funding_oi (timestamp, open_interest)
            VALUES (to_timestamp($1 / 1000.0), $2)
            ON CONFLICT (timestamp) DO UPDATE SET
                open_interest = EXCLUDED.open_interest
            """,
            [
                (d.get('timestamp', 0), d.get('oi', 0))
                for d in batch
            ],
        )

    async def _insert_liquidations(self, conn: asyncpg.Connection, batch: list[dict]) -> None:
        """Insert liquidation records."""
        await conn.executemany(
            """
            INSERT INTO raw_liquidations (timestamp, long_liq_usd, short_liq_usd, liq_ratio)
            VALUES (to_timestamp($1 / 1000.0), $2, $3, $4)
            ON CONFLICT (timestamp) DO UPDATE SET
                long_liq_usd = EXCLUDED.long_liq_usd,
                short_liq_usd = EXCLUDED.short_liq_usd
            """,
            [
                (
                    d.get('timestamp', 0),
                    d.get('liq_long', 0),
                    d.get('liq_short', 0),
                    d.get('liq_ratio', 0),
                )
                for d in batch
            ],
        )

    # =========================================================================
    # Session & State Management
    # =========================================================================

    async def insert_session(self, boot_type: str = 'cold') -> int:
        """Record a new bot session. Returns session ID."""
        if not self._pool:
            return -1
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO bot_session_log (started_at, boot_type)
                VALUES (NOW(), $1)
                RETURNING id
                """,
                boot_type,
            )
            return row['id'] if row else -1

    async def end_session(self, session_id: int, reason: str = 'normal_shutdown') -> None:
        """Mark a session as ended."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE bot_session_log
                SET ended_at = NOW(), reason = $1
                WHERE id = $2
                """,
                reason, session_id,
            )

    async def update_portfolio_state(
        self,
        balance: float,
        bracket_index: int,
        total_bets: int,
        total_wins: int,
        consecutive_losses: int,
    ) -> None:
        """Upsert the current portfolio state."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_state 
                    (id, balance, bracket_index, total_bets, total_wins, consecutive_losses, updated_at)
                VALUES (1, $1, $2, $3, $4, $5, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    balance = EXCLUDED.balance,
                    bracket_index = EXCLUDED.bracket_index,
                    total_bets = EXCLUDED.total_bets,
                    total_wins = EXCLUDED.total_wins,
                    consecutive_losses = EXCLUDED.consecutive_losses,
                    updated_at = NOW()
                """,
                balance, bracket_index, total_bets, total_wins, consecutive_losses,
            )

    async def get_portfolio_state(self) -> Optional[dict]:
        """Fetch current portfolio state."""
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM portfolio_state WHERE id = 1")
            return dict(row) if row else None

    # =========================================================================
    # Query Methods (for dashboard)
    # =========================================================================

    async def fetch(self, query: str, *args) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[dict]:
        """Execute a query and return a single row as dict."""
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    def stats(self) -> dict:
        """Return client statistics."""
        pool_stats = {}
        if self._pool:
            pool_stats = {
                'pool_size': self._pool.get_size(),
                'pool_free': self._pool.get_idle_size(),
            }
        return {
            'connected': self.is_connected,
            'insert_count': self._insert_count,
            'error_count': self._error_count,
            **pool_stats,
        }
