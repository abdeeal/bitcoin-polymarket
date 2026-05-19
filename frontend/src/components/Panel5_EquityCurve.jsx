/** Panel 5 — Equity Curve with dark theme */
import { useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const RANGES = [7, 30, 9999]
const RANGE_LABELS = ['1W', '1M', 'ALL']

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <p style={{ color: '#94a3b8' }}>{new Date(d.timestamp).toLocaleDateString()}</p>
      <p style={{ color: '#00f5d4', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>${d.balance?.toFixed(2)}</p>
    </div>
  )
}

export default function Panel5_EquityCurve({ equityData }) {
  const [range, setRange] = useState(1)
  if (!equityData) return <div className="chart-panel"><div className="panel-loading">Loading chart...</div></div>

  const { equity_curve = [] } = equityData

  const cutoff = Date.now() - RANGES[range] * 864e5
  const data = equity_curve
    .filter(d => new Date(d.timestamp).getTime() > cutoff)
    .map(d => ({ ...d, timestamp: d.timestamp }))

  const startBal = data[0]?.balance ?? 10          // true starting balance (pre-trade point)
  const endBal   = data[data.length - 1]?.balance ?? 10
  const isUp = endBal >= startBal

  return (
    <div className="chart-panel">
      <div className="chart-panel-header">
        <span className="panel-title">Equity Curve</span>
        <div className="toggle-group">
          {RANGE_LABELS.map((l, i) => (
            <button key={l} className={range === i ? 'active' : ''} onClick={() => setRange(i)}>{l}</button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={isUp ? '#00f5d4' : '#ff6b6b'} stopOpacity={0.25} />
              <stop offset="95%" stopColor={isUp ? '#00f5d4' : '#ff6b6b'} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={t => new Date(t).toLocaleDateString('en', { month:'short', day:'numeric' })}
            tick={{ fill: '#64748b', fontSize: 10 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={v => `$${v.toFixed(1)}`}
            tick={{ fill: '#64748b', fontSize: 10 }}
            width={50}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={startBal} stroke="rgba(255,255,255,0.1)" strokeDasharray="4 4" />
          <Area
            type="monotone"
            dataKey="balance"
            stroke={isUp ? '#00f5d4' : '#ff6b6b'}
            strokeWidth={2}
            fill="url(#eqGrad)"
            dot={false}
            animationDuration={800}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
