"""
Reset Database — Clears all data and resets portfolio to initial state ($10 balance).
Run: python reset_database.py
"""
import asyncio
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncpg


TABLES_TO_TRUNCATE = [
    "bet_log",
    "model_predictions",
    "features_5m",
    "raw_candles_1m",
    "raw_orderbook_snapshots",
    "raw_trades",
    "raw_funding_oi",
    "raw_liquidations",
    "cooling_down_log",
    "bot_session_log",
    "portfolio_state",
]


async def reset_database():
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    db   = os.getenv("TIMESCALE_DB", "btc_prediction")
    user = os.getenv("TIMESCALE_USER", "btcbot")
    pw   = os.getenv("TIMESCALE_PASSWORD", "btcbot_secret")

    print(f"Connecting to postgresql://{user}@{host}:{port}/{db} ...")
    conn = await asyncpg.connect(host=host, port=port, database=db, user=user, password=pw)

    print("\n--- Truncating all tables ---")
    for table in TABLES_TO_TRUNCATE:
        try:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
            print(f"  [OK] {table:35s} — {count} rows deleted")
        except Exception as e:
            print(f"  [WARN] {table:33s} — {type(e).__name__}: {e}")

    print("\n--- Resetting portfolio state to $10.00 ---")
    await conn.execute("""
        INSERT INTO portfolio_state (id, balance, bracket_index, total_bets, total_wins, consecutive_losses, updated_at)
        VALUES (1, 10.00, 0, 0, 0, 0, NOW())
        ON CONFLICT (id) DO UPDATE SET
            balance = 10.00,
            bracket_index = 0,
            total_bets = 0,
            total_wins = 0,
            consecutive_losses = 0,
            updated_at = NOW()
    """)
    print("  [OK] portfolio_state reset: balance=10.00, bracket_index=0, all counters=0")

    # Verify
    print("\n--- Verification ---")
    for table in TABLES_TO_TRUNCATE:
        try:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            status = "[OK] empty" if count == 0 else f"[FAIL] {count} rows remaining"
            if table == "portfolio_state":
                row = await conn.fetchrow("SELECT balance, total_bets FROM portfolio_state WHERE id = 1")
                status = f"[OK] balance=${row['balance']}, bets={row['total_bets']}"
            print(f"  {table:35s} — {status}")
        except Exception as e:
            print(f"  {table:35s} — ERROR: {e}")

    await conn.close()
    print("\n[DONE] Database reset complete! Ready for fresh start with $10.00 balance.")


if __name__ == "__main__":
    asyncio.run(reset_database())
