/** Panel 1 — Bot Status & Uptime */
export default function Panel1_Status({ live, status }) {
  // Prefer WebSocket live data, fall back to REST summary.status
  const wsState = live?.bot_state
  const restState = status?.bot_state

  const state = wsState ?? restState ?? 'BOOT'
  const uptime = live?.uptime_formatted ?? status?.uptime_formatted ?? '0m'
  const cdActive = live?.cooling_down_active ?? false
  const cdRemain = live?.cooling_down_remaining_seconds ?? 0
  const lastBet = live?.last_bet ?? {}

  const stateColor = { ACTIVE: '#22c55e', COOLING_DOWN: '#f59e0b', BOOT: '#6366f1', PAUSED: '#ef4444' }
  const dot = stateColor[state] ?? '#6b7280'

  const fmtRemain = (s) => `${Math.floor(s / 60)}m ${s % 60}s`

  return (
    <div className="panel panel-status">
      <div className="status-row">
        <span className="status-dot" style={{ background: dot }} />
        <span className="status-label">BOT {state}</span>
        <span className="status-divider">|</span>
        <span>Uptime: {uptime}</span>
      </div>
      {cdActive && (
        <div className="cooling-badge">
          Cooling Down — {fmtRemain(cdRemain)} remaining
        </div>
      )}
      {lastBet?.timestamp && (
        <div className="last-bet-row">
          Last bet: {new Date(lastBet.timestamp).toLocaleTimeString()}
          &nbsp;·&nbsp;{lastBet.direction}
          &nbsp;·&nbsp;{lastBet.outcome ? '✅ Won' : '❌ Lost'}
        </div>
      )}
    </div>
  )
}
