/** Panel 2 — Balance Overview */
export default function Panel2_Balance({ balance }) {
  if (!balance) return <div className="panel panel-loading">Loading…</div>
  const { starting_balance: start, current_balance: curr,
          total_growth_usd: grow, total_growth_pct: pct,
          highest_balance: hi, lowest_balance: lo } = balance
  const pos = grow >= 0

  return (
    <div className="panel panel-balance">
      <h3 className="panel-title">Balance</h3>
      <div className="balance-cards">
        <div className="bal-card"><span className="bal-label">Starting</span><span className="bal-val">${start?.toFixed(2)}</span></div>
        <div className="bal-card accent"><span className="bal-label">Current</span><span className="bal-val">${curr?.toFixed(2)}</span></div>
        <div className={`bal-card ${pos ? 'green' : 'red'}`}>
          <span className="bal-label">Growth</span>
          <span className="bal-val">{pos ? '+' : ''}{grow?.toFixed(2)} ({pos ? '+' : ''}{pct?.toFixed(1)}%)</span>
        </div>
        <div className="bal-card"><span className="bal-label">Highest</span><span className="bal-val green">${hi?.toFixed(2)}</span></div>
        <div className="bal-card"><span className="bal-label">Lowest</span><span className="bal-val red">${lo?.toFixed(2)}</span></div>
      </div>
    </div>
  )
}
