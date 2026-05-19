/** Panel 4 — Period Gain/Loss */
import { useState } from 'react'

export default function Panel4_PeriodGain({ period }) {
  const [unit, setUnit] = useState('usd')
  if (!period) return <div className="panel-loading">Loading...</div>

  const startBal = 10
  const items = [
    { label: 'Today',     usd: period.pnl_today_usd,  pct: (period.pnl_today_usd  / startBal * 100) },
    { label: 'This Week', usd: period.pnl_week_usd,   pct: (period.pnl_week_usd   / startBal * 100) },
    { label: 'This Month',usd: period.pnl_month_usd,  pct: (period.pnl_month_usd  / startBal * 100) },
  ]

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <span className="panel-title">Period Performance</span>
        <div className="toggle-group">
          <button className={unit==='usd'?'active':''} onClick={()=>setUnit('usd')}>USD</button>
          <button className={unit==='pct'?'active':''} onClick={()=>setUnit('pct')}>%</button>
        </div>
      </div>
      <div className="period-cards">
        {items.map(it => {
          const val = unit === 'usd' ? it.usd : it.pct
          const pos = val >= 0
          return (
            <div key={it.label} className="period-card">
              <span className="period-label">{it.label}</span>
              <span className={`period-val ${pos ? 'green' : 'red'}`}>
                {pos ? '+' : ''}{unit==='usd' ? `$${val?.toFixed(2)}` : `${val?.toFixed(1)}%`}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
