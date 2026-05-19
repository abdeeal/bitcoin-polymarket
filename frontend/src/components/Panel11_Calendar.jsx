/** Panel 11 — Performance Calendar */
import { useState } from 'react'

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const DAYS   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']

export default function Panel11_Calendar({ calendar, onMonthChange }) {
  const tzDate = new Date(new Date().toLocaleString("en-US", {timeZone: "Asia/Jakarta"}));
  const [year, setYear] = useState(tzDate.getFullYear())
  const [month, setMonth] = useState(tzDate.getMonth() + 1)

  const navigate = (delta) => {
    let m = month + delta
    let y = year
    if (m < 1) { m = 12; y-- }
    if (m > 12) { m = 1; y++ }
    setMonth(m); setYear(y)
    onMonthChange?.(y, m)
  }

  const daysInMonth = new Date(year, month, 0).getDate()
  const firstDow = new Date(year, month - 1, 1).getDay()

  const dataByDate = {}
  if (calendar) {
    for (const d of calendar) {
      dataByDate[d.date.slice(0, 10)] = d
    }
  }

  const cells = []
  for (let i = 0; i < firstDow; i++) cells.push(null)
  for (let d = 1; d <= daysInMonth; d++) {
    const dateStr = `${year}-${String(month).padStart(2,'0')}-${String(d).padStart(2,'0')}`
    cells.push({ day: d, data: dataByDate[dateStr] ?? null, dateStr })
  }

  const todayDateStr = `${tzDate.getFullYear()}-${String(tzDate.getMonth() + 1).padStart(2, '0')}-${String(tzDate.getDate()).padStart(2, '0')}`;
  const isToday = (dateStr) => dateStr === todayDateStr

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <span className="panel-title">Calendar - {MONTHS[month-1]} {year}</span>
        <div className="toggle-group">
          <button onClick={() => navigate(-1)}>{'\u25C0'}</button>
          <button onClick={() => navigate(1)}>{'\u25B6'}</button>
        </div>
      </div>
      <div className="cal-grid">
        {DAYS.map(d => <div key={d} className="cal-header-cell">{d}</div>)}
        {cells.map((cell, i) => {
          if (cell === null) return <div key={`e${i}`} className="cal-cell empty" />
          const { day, data, dateStr } = cell
          const today = isToday(dateStr)
          const color = data ? (data.is_profit_day ? '#00f5d4' : '#ff6b6b') : 'transparent'
          const bgClass = data ? (data.is_profit_day ? 'profit-day' : 'loss-day') : ''
          return (
            <div
              key={dateStr}
              className={`cal-cell ${today ? 'today' : ''} ${bgClass}`}
              title={data ? `${data.total_bets} bets | PnL ${data.total_pnl >= 0 ? '+' : ''}$${data.total_pnl?.toFixed(2)} | WR ${data.win_rate_pct?.toFixed(1)}%` : ''}
            >
              <span className="cal-day-num">{day}</span>
              {data && (
                <div className="cal-pnl" style={{ color }}>
                  {data.total_pnl >= 0 ? '+' : ''}{data.total_pnl?.toFixed(2)}
                  <div style={{ fontSize: '9px', marginTop: '2px', color: 'rgba(255,255,255,0.6)', fontWeight: '500' }}>
                    {data.total_bets} trades
                  </div>
                </div>
              )}
              {!data && <span className="cal-empty-dot">{'\u00B7'}</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}
