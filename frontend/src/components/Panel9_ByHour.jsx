/** Panel 9 — Performance by Hour (UTC) */
import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'

const HourTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <p>Jam {d.hour}:00 WIB</p>
      <p>{d.total_bets} bets</p>
      <p className={d.total_pnl >= 0 ? 'green' : 'red'}>PnL: {d.total_pnl >= 0 ? '+' : ''}${d.total_pnl?.toFixed(2)}</p>
      <p>WR: {d.win_rate_pct?.toFixed(1)}%</p>
    </div>
  )
}

export default function Panel9_ByHour({ byHour }) {
  const [metric, setMetric] = useState('pnl')

  if (byHour === undefined || byHour === null) return <div className="panel-loading">Loading...</div>
  if (!byHour.length) return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Performance by Hour (WIB)</div>
      <div className="panel-empty">No trades yet</div>
    </div>
  )

  return (
    <div className="chart-section">
      <div className="chart-section-header">
        <span className="panel-title">Performance by Hour (WIB)</span>
        <div className="toggle-group">
          <button className={metric==='pnl'?'active':''} onClick={()=>setMetric('pnl')}>PnL $</button>
          <button className={metric==='wr'?'active':''} onClick={()=>setMetric('wr')}>Win %</button>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={byHour} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="hour" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} tickLine={false} />
          <YAxis tickFormatter={v => metric === 'pnl' ? `$${v}` : `${v}%`} tick={{ fill: '#64748b', fontSize: 10 }} width={45} axisLine={false} tickLine={false} />
          <Tooltip content={<HourTooltip />} cursor={{ fill: 'rgba(255, 255, 255, 0.1)' }} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,)" />
          <Bar dataKey={metric === 'pnl' ? 'total_pnl' : 'win_rate_pct'} radius={[3,3,0,0]}>
            {byHour.map((d, i) => (
              <Cell key={i} fill={
                metric === 'pnl'
                  ? (d.total_pnl >= 0 ? '#00f5d4' : '#ff6b6b')
                  : (d.win_rate_pct >= 53 ? '#00f5d4' : '#ffd43b')
              } />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
