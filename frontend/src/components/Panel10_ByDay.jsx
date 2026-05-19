/** Panel 10 — Performance by Day of Week */
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const DayTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <p>{d.day_name}</p>
      <p>{d.total_bets} bets</p>
      <p className={d.total_pnl >= 0 ? 'green' : 'red'}>{d.total_pnl >= 0 ? '+' : ''}${d.total_pnl?.toFixed(2)}</p>
      <p>WR: {d.win_rate_pct?.toFixed(1)}%</p>
    </div>
  )
}

export default function Panel10_ByDay({ byDay }) {
  if (byDay === undefined || byDay === null) return <div className="panel-loading">Loading...</div>
  if (!byDay.length) return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Performance by Day</div>
      <div className="panel-empty">No trades yet</div>
    </div>
  )

  return (
    <div className="chart-section">
      <div className="chart-section-header">
        <span className="panel-title">Performance by Day</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={byDay} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="day_name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.06)' }} tickLine={false} />
          <YAxis tickFormatter={v => `$${v}`} tick={{ fill: '#64748b', fontSize: 11 }} width={45} axisLine={false} tickLine={false} />
          <Tooltip content={<DayTooltip />} cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }} />
          <Bar dataKey="total_pnl" radius={[3,3,0,0]}>
            {byDay.map((d, i) => (
              <Cell key={i} fill={d.total_pnl >= 0 ? '#00f5d4' : '#ff6b6b'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
