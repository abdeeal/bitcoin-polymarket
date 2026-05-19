import asyncio, asyncpg
async def fix():
    pool = await asyncpg.create_pool('postgresql://btcbot:your_db_password_here@localhost:5432/btc_prediction')
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE cooling_down_log 
            SET ended_at = NOW(), duration_sec = EXTRACT(EPOCH FROM (NOW() - started_at))
            WHERE ended_at IS NULL AND EXTRACT(EPOCH FROM (NOW() - started_at)) >= 1800;
        """)
        print("Fixed hanging cooling down logs.")
asyncio.run(fix())
