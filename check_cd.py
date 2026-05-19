import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def check():
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    db = os.getenv("TIMESCALE_DB", "btc_prediction")
    user = os.getenv("TIMESCALE_USER", "btcbot")
    password = os.getenv("TIMESCALE_PASSWORD", "btcbot_secret")
    conn = await asyncpg.connect(host=host, port=port, database=db, user=user, password=password)
    rows = await conn.fetch("SELECT * FROM cooling_down_log")
    for r in rows:
        print(dict(r))
    await conn.close()

asyncio.run(check())
