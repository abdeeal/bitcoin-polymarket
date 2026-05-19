/** Panel 3 — Total PnL */
export default function Panel3_TotalPnl({ pnl }) {
  if (!pnl) return <div className="panel panel-loading">Loading…</div>
  const pos = pnl.total_pnl_gross >= 0
  return (
    <div className="panel">
      <h3 className="panel-title">Total PnL</h3>
      <div className={`pnl-main ${pos ? 'green' : 'red'}`}>
        {pos ? '+' : ''}{pnl.total_pnl_gross?.toFixed(2)} ({pos ? '+' : ''}{pnl.total_pnl_pct?.toFixed(1)}%)
      </div>
      <div className="pnl-detail-row"><span>Gross</span><span>${pnl.total_pnl_gross?.toFixed(2)}</span></div>
      <div className="pnl-detail-row"><span>Fees paid</span><span className="red">-${pnl.total_fees?.toFixed(2)}</span></div>
      <div className="pnl-detail-row total"><span>Net</span><span className={pos ? 'green' : 'red'}>${pnl.total_pnl_net?.toFixed(2)}</span></div>
    </div>
  )
}
