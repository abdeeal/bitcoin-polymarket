/** StatCards — Top 4 stat cards */
export default function StatCards({ balance, pnl, streaks, period, live }) {
  // Current balance: prefer WebSocket (updates every 5s), fallback to REST (30s)
  const curr     = live?.current_balance ?? balance?.current_balance ?? 10.0
  // Starting balance: always the fixed starting capital (10.0),
  // NOT balance_before of first bet (which equals 10.0 anyway but could differ).
  // growth = current - starting
  const startBal  = balance?.starting_balance ?? 10.0
  const growthUsd = curr - startBal
  const growthPct = startBal > 0 ? (growthUsd / startBal * 100) : 0
  const growthPos = growthPct >= 0

  // Today PnL: sum of PnL of bets placed today (UTC+7)
  // period.pnl_today_usd is already filtered by CURRENT_DATE in backend
  const todayPnl = live?.today_pnl ?? period?.pnl_today_usd ?? 0
  const todayPos = todayPnl >= 0

  const winRate    = live?.win_rate ?? streaks?.win_rate ?? 0
  const winRatePct = (winRate * 100).toFixed(1)
  const prevWr     = 50
  const wrDelta    = (winRate * 100 - prevWr).toFixed(1)

  const hcWinRate = live?.high_conf_win_rate ?? streaks?.high_conf_win_rate ?? 0
  const hcWrPct   = (hcWinRate * 100).toFixed(1)

  return (
    <div className="stat-cards">
      {/* Card 1 — Balance */}
      <div className="stat-card card-balance">
        <div className="sc-label">Current Balance</div>
        <div className="sc-value">${curr.toFixed(2)}</div>
        <div className={`sc-sub ${growthPos ? 'positive' : 'negative'}`}>
          {growthPos ? '↑' : '↓'}{Math.abs(growthPct).toFixed(2)}%
          <span style={{ marginLeft: 4, opacity: 0.7 }}>
            ({growthPos ? '+' : ''}${growthUsd.toFixed(2)})
          </span>
        </div>
      </div>

      {/* Card 2 — Daily PnL */}
      <div className="stat-card card-pnl">
        <div className="sc-label">Daily PnL</div>
        <div className="sc-value" style={{ color: todayPos ? 'var(--green)' : 'var(--red)' }}>
          {todayPos ? '+' : '-'}${Math.abs(todayPnl).toFixed(2)}
        </div>
        <div className="sc-sub neutral">Today (WIB)</div>
      </div>

      {/* Card 3 — Win Rate */}
      <div className="stat-card card-winrate">
        <div className="sc-label">Win Rate</div>
        <div className="sc-value">{winRatePct}%</div>
        <div className={`sc-sub ${parseFloat(wrDelta) >= 0 ? 'positive' : 'negative'}`}>
          {parseFloat(wrDelta) >= 0 ? '↑' : '↓'}{Math.abs(parseFloat(wrDelta)).toFixed(1)}% vs 50%
        </div>
      </div>

      {/* Card 4 — High Conf Win Rate */}
      <div className="stat-card card-factor">
        <div className="sc-label">High Conf WR</div>
        <div className="sc-value">{hcWrPct}%</div>
        <div className={`sc-sub ${hcWinRate >= 0.53 ? 'positive' : 'negative'}`}>
          ⚡︎ {hcWinRate > 0 ? 'High Edge Bets' : 'No data'}
        </div>
      </div>
    </div>
  )
}
