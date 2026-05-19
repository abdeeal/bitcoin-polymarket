/** Panel 8 — Win Rate by Side */
export default function Panel8_WinRateSide({ bySide }) {
  if (bySide === undefined || bySide === null) return <div className="panel-loading">Loading...</div>
  if (!bySide.length) return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Win Rate by Side</div>
      <div className="panel-empty">No trades yet</div>
    </div>
  )

  return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Win Rate by Side</div>
      {bySide.map(r => {
        const wr = r.win_rate_pct ?? 0
        const isBuy = r.side === 'BUY'
        return (
          <div key={r.side} className="wr-row">
            <span className={`wr-label ${isBuy ? 'badge-up' : 'badge-down'}`}>{isBuy ? '\u25B2 BUY' : '\u25BC SELL'}</span>
            <div className="wr-bar-wrap">
              <div className="wr-bar" style={{ width: `${wr}%`, background: isBuy ? '#00f5d4' : '#4cc9f0' }} />
            </div>
            <span className="wr-pct accent">{wr.toFixed(1)}%</span>
            <span className="wr-sub">({r.wins}/{r.entries})</span>
          </div>
        )
      })}
    </div>
  )
}
