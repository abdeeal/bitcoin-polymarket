import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def fix_balances():
    host = os.getenv("TIMESCALE_HOST", "localhost")
    port = int(os.getenv("TIMESCALE_PORT", "5432"))
    db = os.getenv("TIMESCALE_DB", "btc_prediction")
    user = os.getenv("TIMESCALE_USER", "btcbot")
    password = os.getenv("TIMESCALE_PASSWORD", "btcbot_secret")

    try:
        conn = await asyncpg.connect(host=host, port=port, database=db, user=user, password=password)
        
        # Get all bets ordered by timestamp ASC
        rows = await conn.fetch("SELECT id, pnl FROM bet_log WHERE outcome IS NOT NULL ORDER BY timestamp ASC")
        
        current_balance = 10.00
        
        for row in rows:
            bet_id = row['id']
            pnl = float(row['pnl'])
            
            balance_before = current_balance
            balance_after = current_balance + pnl
            
            await conn.execute("""
                UPDATE bet_log 
                SET balance_before = $1, balance_after = $2
                WHERE id = $3
            """, balance_before, balance_after, bet_id)
            
            current_balance = balance_after

        # Also update portfolio_state
        await conn.execute("UPDATE portfolio_state SET balance = $1 WHERE id = 1", current_balance)
        print(f"Database sync complete. True portfolio balance should be: {current_balance:.2f}")
        
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fix_balances())
