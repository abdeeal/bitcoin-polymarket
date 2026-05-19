"""Create the 3 missing tables: cooling_down_log, bot_session_log, portfolio_state"""
import asyncio
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncpg


async def run():
    conn = await asyncpg.connect(
        host=os.getenv("TIMESCALE_HOST", "localhost"),
        port=int(os.getenv("TIMESCALE_PORT", "5432")),
        database=os.getenv("TIMESCALE_DB", "btc_prediction"),
        user=os.getenv("TIMESCALE_USER", "btcbot"),
        password=os.getenv("TIMESCALE_PASSWORD", "btcbot_secret"),
    )
    print("Connected.")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS cooling_down_log (
            id               SERIAL PRIMARY KEY,
            started_at       TIMESTAMPTZ NOT NULL,
            ended_at         TIMESTAMPTZ,
            trigger_reason   TEXT NOT NULL DEFAULT '2_consecutive_losses',
            duration_sec     INTEGER,
            balance_at_start NUMERIC(10,2),
            bracket_at_start INTEGER,
            candles_missed   INTEGER DEFAULT 0
        )
    """)
    print("[OK] cooling_down_log")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_session_log (
            id         SERIAL PRIMARY KEY,
            started_at TIMESTAMPTZ NOT NULL,
            ended_at   TIMESTAMPTZ,
            boot_type  TEXT,
            reason     TEXT
        )
    """)
    print("[OK] bot_session_log")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id                 INTEGER PRIMARY KEY DEFAULT 1,
            balance            NUMERIC(10,2) NOT NULL DEFAULT 10.00,
            bracket_index      INTEGER NOT NULL DEFAULT 0,
            total_bets         INTEGER NOT NULL DEFAULT 0,
            total_wins         INTEGER NOT NULL DEFAULT 0,
            consecutive_losses INTEGER NOT NULL DEFAULT 0,
            updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (id = 1)
        )
    """)
    print("[OK] portfolio_state")

    await conn.execute("""
        INSERT INTO portfolio_state (id, balance)
        VALUES (1, 10.00)
        ON CONFLICT (id) DO NOTHING
    """)
    print("[OK] portfolio_state seeded with $10.00 starting balance")

    rows = await conn.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' ORDER BY table_name"
    )
    print("\nAll tables in database:")
    for r in rows:
        print(f"  - {r['table_name']}")

    await conn.close()
    print("\nDone! Restart uvicorn to apply.")


if __name__ == "__main__":
    asyncio.run(run())
