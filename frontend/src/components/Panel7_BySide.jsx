/** Panel 7 — Performance by Side */
export default function Panel7_BySide({ bySide }) {
  if (bySide === undefined || bySide === null) return <div className="panel-loading">Loading...</div>
  if (!bySide.length) return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Performance by Side</div>
      <div className="panel-empty">No trades yet</div>
    </div>
  )

  const total = bySide.reduce((s, r) => s + r.entries, 0)
  const totalPnl = bySide.reduce((s, r) => s + r.total_pnl, 0)

  return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Performance by Side</div>
      <table className="side-table">
        <thead><tr><th>Side</th><th>Entries</th><th>Total PnL</th><th>Avg PnL</th></tr></thead>
        <tbody>
          {bySide.map(r => (
            <tr key={r.side}>
              <td><span className={r.side === 'BUY' ? 'badge-up' : 'badge-down'}>{r.side === 'BUY' ? '\u25B2 BUY' : '\u25BC SELL'}</span></td>
              <td>{r.entries}</td>
              <td className={r.total_pnl >= 0 ? 'green' : 'red'}>{r.total_pnl >= 0 ? '+' : ''}${r.total_pnl.toFixed(2)}</td>
              <td className={r.avg_pnl >= 0 ? 'green' : 'red'}>{r.avg_pnl >= 0 ? '+' : ''}${r.avg_pnl.toFixed(2)}</td>
            </tr>
          ))}
          <tr className="total-row">
            <td>Total</td><td>{total}</td>
            <td className={totalPnl >= 0 ? 'green' : 'red'}>{totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}</td>
            <td>-</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
