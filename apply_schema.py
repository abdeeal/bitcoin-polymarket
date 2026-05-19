"""
One-shot script to apply schema.sql to TimescaleDB.
Run: python apply_schema.py
"""
import asyncio
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncpg


async def apply_schema():
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    db   = os.getenv("TIMESCALE_DB", "btc_prediction")
    user = os.getenv("TIMESCALE_USER", "btcbot")
    pw   = os.getenv("TIMESCALE_PASSWORD", "btcbot_secret")

    print(f"Connecting to postgresql://{user}@{host}:{port}/{db} ...")

    conn = await asyncpg.connect(host=host, port=port, database=db, user=user, password=pw)

    with open("src/db/schema.sql", "r", encoding="utf-8") as f:
        sql = f.read()

    # Split by statement and run each individually (asyncpg doesn't support multi-statement)
    statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]

    ok = 0
    skipped = 0
    errors = []
    for stmt in statements:
        try:
            await conn.execute(stmt)
            ok += 1
        except asyncpg.exceptions.DuplicateTableError:
            skipped += 1
        except asyncpg.exceptions.DuplicateObjectError:
            skipped += 1
        except asyncpg.exceptions.UniqueViolationError:
            skipped += 1
        except Exception as e:
            errors.append(f"WARN: {type(e).__name__}: {e}\n  SQL: {stmt[:80]}...")

    await conn.close()

    print(f"\nSchema applied: {ok} statements OK, {skipped} already existed")
    if errors:
        print(f"WARNINGS: {len(errors)} non-fatal:")
        for err in errors:
            print(" ", err)
    else:
        print("OK - No errors -- all tables ready!")

    print("\nVerifying tables...")
    conn2 = await asyncpg.connect(host=host, port=port, database=db, user=user, password=pw)
    rows = await conn2.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
    )
    await conn2.close()
    for r in rows:
        print(f"  [OK] {r['table_name']}")


if __name__ == "__main__":
    asyncio.run(apply_schema())
