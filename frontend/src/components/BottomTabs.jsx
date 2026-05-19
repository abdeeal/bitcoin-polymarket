/** BottomTabs — Tabbed section for Recent Trades, Statistics, Charts, Calendar */
import { useState } from 'react'
import Panel6_Streaks from './Panel6_Streaks'
import Panel7_BySide from './Panel7_BySide'
import Panel8_WinRateSide from './Panel8_WinRateSide'
import Panel9_ByHour from './Panel9_ByHour'
import Panel10_ByDay from './Panel10_ByDay'
import Panel11_Calendar from './Panel11_Calendar'
import Panel4_PeriodGain from './Panel4_PeriodGain'

const TABS = [
  { key: 'stats', label: 'Statistics' },
  { key: 'charts', label: 'Charts' },
  { key: 'calendar', label: 'Calendar' },
]

export default function BottomTabs({ bySide, byHour, byDay, calendar, streaks, period, onMonthChange }) {
  const [tab, setTab] = useState('stats')

  return (
    <div className="bottom-section">
      <div className="bottom-tabs">
        {TABS.map(t => (
          <div
            key={t.key}
            className={`bottom-tab ${tab === t.key ? 'active' : ''}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </div>
        ))}
      </div>

      <div className="bottom-content">
        {tab === 'stats' && (
          <div>
            {/* Trade Statistics Grid */}
            <Panel6_Streaks streaks={streaks} />

            {/* Side Performance */}
            <div className="side-performance-grid">
              <div>
                <Panel7_BySide bySide={bySide} />
              </div>
              <div>
                <Panel8_WinRateSide bySide={bySide} />
              </div>
            </div>

            {/* Period Performance */}
            <div style={{ marginTop: 16 }}>
              <Panel4_PeriodGain period={period} />
            </div>
          </div>
        )}

        {tab === 'charts' && (
          <div>
            <Panel9_ByHour byHour={byHour} />
            <div style={{ marginTop: 16 }}>
              <Panel10_ByDay byDay={byDay} />
            </div>
          </div>
        )}

        {tab === 'calendar' && (
          <Panel11_Calendar calendar={calendar} onMonthChange={onMonthChange} />
        )}
      </div>
    </div>
  )
}
