import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import useWebSocket from './hooks/useWebSocket'
import Sidebar from './components/Sidebar'
import StatCards from './components/StatCards'
import Panel5_EquityCurve from './components/Panel5_EquityCurve'
import PhasePanel from './components/PhasePanel'
import BottomTabs from './components/BottomTabs'
import TradeHistory from './components/TradeHistory'
import FeaturesMonitor from './components/FeaturesMonitor'
import './App.css'

const API = '/api'
const fetcher = (url) => fetch(API + url).then(r => r.json())

export default function App() {
  const [live, setLive] = useState(null)
  const [calYear, setCalYear] = useState(new Date().getFullYear())
  const [calMonth, setCalMonth] = useState(new Date().getMonth() + 1)
  const [activePage, setActivePage] = useState('overview')
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  // WebSocket — real-time tick updates (every 5s)
  const onWsMessage = useCallback((data) => setLive(data), [])
  useWebSocket(onWsMessage)

  // REST — historical data (auto-refresh every 30s)
  const qOpts = { staleTime: 30_000, refetchInterval: 30_000 }
  const { data: summary }  = useQuery({ queryKey: ['summary'],        queryFn: () => fetcher('/stats/summary'),                       ...qOpts })
  const { data: period }   = useQuery({ queryKey: ['period'],         queryFn: () => fetcher('/stats/period'),                        ...qOpts })
  const { data: equity }   = useQuery({ queryKey: ['equity'],         queryFn: () => fetcher('/stats/equity-curve?days=30'),          ...qOpts })
  const { data: streaks }  = useQuery({ queryKey: ['streaks'],        queryFn: () => fetcher('/stats/streaks'),                       ...qOpts })
  const { data: bySide }   = useQuery({ queryKey: ['by-side'],        queryFn: () => fetcher('/stats/by-side'),                       ...qOpts })
  const { data: byHour }   = useQuery({ queryKey: ['by-hour'],        queryFn: () => fetcher('/stats/by-hour'),                       ...qOpts })
  const { data: byDay }    = useQuery({ queryKey: ['by-day'],         queryFn: () => fetcher('/stats/by-day'),                        ...qOpts })
  const { data: prediction } = useQuery({ queryKey: ['prediction'],   queryFn: () => fetcher('/prediction/latest'), staleTime: 10_000, refetchInterval: 10_000 })
  const { data: calendar, refetch: refetchCal } = useQuery({
    queryKey: ['calendar', calYear, calMonth],
    queryFn: () => fetcher(`/stats/by-calendar?year=${calYear}&month=${calMonth}`),
    ...qOpts,
  })

  const handleMonthChange = (y, m) => {
    setCalYear(y); setCalMonth(m)
    setTimeout(() => refetchCal(), 100)
  }

  const pageTitle = {
    overview:    'Analytics Overview',
    history:     'Trade History',
    performance: 'Performance',
    features:    'Features Monitoring',
    settings:    'Settings',
  }[activePage] ?? 'Analytics Overview'

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <Sidebar
        live={live}
        status={summary?.status}
        predictionData={prediction}
        activePage={activePage}
        onPageChange={(page) => { setActivePage(page); setIsMobileMenuOpen(false); }}
        isOpen={isMobileMenuOpen}
        onClose={() => setIsMobileMenuOpen(false)}
      />

      {/* ── Mobile Overlay ── */}
      <div 
        className={`sidebar-overlay ${isMobileMenuOpen ? 'open' : ''}`} 
        onClick={() => setIsMobileMenuOpen(false)}
      />

      {/* ── Main Content ── */}
      <div className="main-content">
        <div className="main-header">
          <div className="main-header-left">
            <button className="mobile-menu-btn" onClick={() => setIsMobileMenuOpen(true)}>☰</button>
            <h1>{pageTitle}</h1>
          </div>
          <div className="main-header-right">
            <div className="header-badge">
              <span className={`ws-dot ${live ? 'connected' : 'disconnected'}`} />
              {live ? 'Live Data' : 'Connecting...'}
            </div>
          </div>
        </div>

        <div className="main-scroll">
          {/* ── Overview Page ── */}
          {activePage === 'overview' && (
            <>
              {/* Row 1 — Stat Cards */}
              <StatCards
                balance={summary?.balance}
                pnl={summary?.pnl}
                streaks={streaks}
                period={period}
                live={live}
              />

              {/* Row 2 — Equity Curve + Phase Panel */}
              <div className="mid-row">
                <Panel5_EquityCurve equityData={equity} />
                <PhasePanel
                  streaks={streaks}
                  period={period}
                  balance={summary?.balance}
                  live={live}
                />
              </div>

              {/* Row 3 — Tabbed Bottom Section */}
              <BottomTabs
                bySide={bySide}
                byHour={byHour}
                byDay={byDay}
                calendar={calendar}
                streaks={streaks}
                period={period}
                onMonthChange={handleMonthChange}
              />
            </>
          )}

          {/* ── History Page ── */}
          {activePage === 'history' && (
            <TradeHistory />
          )}

          {/* ── Performance Page (reuse bottom section standalone) ── */}
          {activePage === 'performance' && (
            <BottomTabs
              bySide={bySide}
              byHour={byHour}
              byDay={byDay}
              calendar={calendar}
              streaks={streaks}
              period={period}
              onMonthChange={handleMonthChange}
            />
          )}

          {/* ── Settings Page ── */}
          {activePage === 'settings' && (
            <div className="panel" style={{ padding: 32, textAlign: 'center', color: 'var(--muted)' }}>
              Settings coming soon
            </div>
          )}

          {/* ── Features Monitoring Page ── */}
          {activePage === 'features' && (
            <FeaturesMonitor />
          )}
        </div>
      </div>
    </div>
  )
}
