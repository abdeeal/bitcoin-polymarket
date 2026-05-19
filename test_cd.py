import asyncio, asyncpg, os
async def test():
    pool = await asyncpg.create_pool('postgresql://btcbot:btcbot_secret@localhost:5432/btc_prediction')
    async with pool.acquire() as conn:
        print(await conn.fetch('SELECT started_at, ended_at, EXTRACT(EPOCH FROM (NOW() - started_at)) as elapsed FROM cooling_down_log WHERE ended_at IS NULL'))
asyncio.run(test())
