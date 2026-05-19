/** PhasePanel — Trading Evaluation Panel (right side of mid row) */
export default function PhasePanel({ streaks, period, balance, live }) {
  const state = live?.bot_state ?? 'ACTIVE'
  const totalBets = streaks?.total_entries ?? 0
  const wins = streaks?.total_wins ?? 0
  const losses = streaks?.total_losses ?? 0
  const winRate = streaks?.win_rate ?? 0
  const winRatePct = (winRate * 100).toFixed(1)
  const consLosses = live?.consecutive_losses ?? 0

  const currBal = live?.current_balance ?? balance?.current_balance ?? 10.0
  const startBal = balance?.starting_balance ?? 10.0
  const growthPct = startBal > 0 ? ((currBal - startBal) / startBal * 100) : 0

  // Period PnL
  const todayPnl = live?.today_pnl ?? period?.pnl_today_usd ?? 0
  const weekPnl = period?.pnl_week_usd ?? 0
  const monthPnl = period?.pnl_month_usd ?? 0

  // Progress metrics
  const targetBets = 50  // validation target
  const betProgress = Math.min(100, (totalBets / targetBets) * 100)
  const targetWR = 53    // 53% target win rate
  const wrProgress = Math.min(100, (parseFloat(winRatePct) / targetWR) * 100)

  return (
    <div className="phase-panel">
      <div className="phase-header">
        <span className="panel-title">Trading Evaluation</span>
        <span className={`phase-badge ${state === 'COOLING_DOWN' ? 'cooling' : 'active'}`}>
          {state === 'COOLING_DOWN' ? 'COOLING' : state}
        </span>
      </div>

      {/* Win Rate Target */}
      <div className="phase-item">
        <div className="phase-item-header">
          <span className="phase-item-label">Win Rate Target (53%)</span>
          <span className="phase-item-value" style={{ color: parseFloat(winRatePct) >= 53 ? 'var(--green)' : 'var(--red)' }}>
            {winRatePct}%
          </span>
        </div>
        <div className="phase-bar">
          <div className="phase-bar-fill" style={{ width: `${wrProgress}%`, background: parseFloat(winRatePct) >= 53 ? 'var(--green)' : 'var(--yellow)' }} />
        </div>
      </div>

      {/* Bets Progress */}
      <div className="phase-item">
        <div className="phase-item-header">
          <span className="phase-item-label">Bets Placed (target: {targetBets})</span>
          <span className="phase-item-value">
            {totalBets} / {targetBets}
          </span>
        </div>
        <div className="phase-bar">
          <div className="phase-bar-fill" style={{ width: `${betProgress}%`, background: 'var(--blue)' }} />
        </div>
      </div>

      {/* Consecutive Losses */}
      <div className="phase-item">
        <div className="phase-item-header">
          <span className="phase-item-label">Consecutive Losses</span>
          <span className="phase-item-value" style={{ color: consLosses >= 2 ? 'var(--red)' : 'var(--text)' }}>
            {consLosses} / 2
          </span>
        </div>
        <div className="phase-bar">
          <div className="phase-bar-fill" style={{ width: `${(consLosses / 2) * 100}%`, background: consLosses >= 2 ? 'var(--red)' : 'var(--yellow)' }} />
        </div>
      </div>

      {/* Growth */}
      <div className="phase-item">
        <div className="phase-item-header">
          <span className="phase-item-label">Portfolio Growth</span>
          <span className="phase-item-value" style={{ color: growthPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {growthPct >= 0 ? '+' : ''}{growthPct.toFixed(1)}%
          </span>
        </div>
        <div className="phase-bar">
          <div className="phase-bar-fill" style={{ width: `${Math.min(100, Math.abs(growthPct))}%`, background: growthPct >= 0 ? 'var(--green)' : 'var(--red)' }} />
        </div>
      </div>

      {/* Period Performance Summary */}
      <div className="phase-item">
        <div className="phase-item-header">
          <span className="phase-item-label">Record (W/L)</span>
          <span className="phase-item-value">
            <span style={{ color: 'var(--green)' }}>{wins}</span>
            <span style={{ color: 'var(--muted)' }}> / </span>
            <span style={{ color: 'var(--red)' }}>{losses}</span>
          </span>
        </div>
      </div>

      {/* Footer Stats */}
      <div className="phase-footer">
        <div className="phase-footer-item">
          <span className="phase-footer-label">Today</span>
          <span className="phase-footer-val" style={{ color: todayPnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {todayPnl >= 0 ? '+' : ''}${todayPnl.toFixed(2)}
          </span>
        </div>
        <div className="phase-footer-item">
          <span className="phase-footer-label">Week</span>
          <span className="phase-footer-val" style={{ color: weekPnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {weekPnl >= 0 ? '+' : ''}${weekPnl.toFixed(2)}
          </span>
        </div>
        <div className="phase-footer-item">
          <span className="phase-footer-label">Month</span>
          <span className="phase-footer-val" style={{ color: monthPnl >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {monthPnl >= 0 ? '+' : ''}${monthPnl.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  )
}
