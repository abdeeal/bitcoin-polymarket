/** Sidebar — Logo, Status, Prediction Card, Navigation */
import { useState, useEffect } from 'react'

export default function Sidebar({ live, status, predictionData, activePage, onPageChange, isOpen, onClose }) {
  const pred = live?.next_prediction ?? predictionData ?? null
  const [countdown, setCountdown] = useState(0)
  const [cdTimer, setCdTimer] = useState(0)

  useEffect(() => {
    if (!pred) {
      const now = Math.floor(Date.now() / 1000)
      const nextTs = Math.ceil(now / 300) * 300
      setCountdown(Math.max(0, nextTs - now))
    } else {
      setCountdown(pred.remaining_seconds ?? 0)
    }
    const id = setInterval(() => setCountdown(p => (p > 0 ? p - 1 : 0)), 1000)
    return () => clearInterval(id)
  }, [pred?.target_time])

  useEffect(() => {
    if (countdown === 0) {
      const t = setTimeout(() => {
        const now = Math.floor(Date.now() / 1000)
        setCountdown(Math.max(0, Math.ceil(now / 300) * 300 - now))
      }, 1000)
      return () => clearTimeout(t)
    }
  }, [countdown])

  useEffect(() => {
    setCdTimer(live?.cooling_down_remaining_seconds ?? 0)
  }, [live?.cooling_down_remaining_seconds])

  useEffect(() => {
    const id = setInterval(() => setCdTimer(p => (p > 0 ? p - 1 : 0)), 1000)
    return () => clearInterval(id)
  }, [])

  // Status
  const wsState = live?.bot_state
  const restState = status?.bot_state
  const state = wsState ?? restState ?? 'BOOT'
  const uptime = live?.uptime_formatted ?? status?.uptime_formatted ?? '0m'
  const cdActive = live?.cooling_down_active ?? false
  const lastBet = live?.last_bet ?? {}
  const stateColors = { ACTIVE: '#00f5d4', COOLING_DOWN: '#ffd43b', BOOT: '#7c3aed', PAUSED: '#ff6b6b' }
  const dot = stateColors[state] ?? '#64748b'

  // Prediction
  const direction = pred?.direction ?? 'SKIP'
  const confidence = pred?.confidence ?? 0
  const rawProb = pred?.raw_prob ?? confidence
  const reason = pred?.reason ?? 'Waiting for data'
  const targetTime = (() => {
    if (pred?.target_time) return pred.target_time
    const now = Math.floor(Date.now() / 1000)
    const d = new Date(Math.ceil(now / 300) * 300 * 1000)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  })()

  // Fix: (confidence * 100).toFixed(2) — NOT confidence.toFixed(2) * 100 (that gives NaN)
  const confidencePct = (confidence * 100).toFixed(2)
  const rawProbPct    = (rawProb    * 100).toFixed(2)
  const barColor = parseFloat(confidencePct) >= 70 ? '#00f5d4' : parseFloat(confidencePct) >= 55 ? '#ffd43b' : '#64748b'
  const fmtCD = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  const fmtRemain = (s) => `${Math.floor(s / 60)}m ${s % 60}s`

  const navItems = [
    { key: 'overview',    icon: '☰', label: 'Overview' },
    { key: 'history',     icon: '⊞', label: 'History' },
    { key: 'performance', icon: '☆', label: 'Performance' },
    { key: 'features',    icon: '모', label: 'Features Status' },
    // { key: 'settings',    icon: '⚑', label: 'Settings' },
  ]

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">B</div>
        <div className="sidebar-logo-text">
          <strong>Quant Opus</strong>
          <small>Terminal v1.3</small>
        </div>
        <button className="sidebar-close-btn" onClick={onClose}>×</button>
      </div>

      {/* Status Badge */}
      <div className="sidebar-status">
        <span className="sidebar-status-dot" style={{ background: dot, boxShadow: `0 0 8px ${dot}` }} />
        <span className="sidebar-status-label">{state}</span>
        <span className="sidebar-uptime">{uptime}</span>
      </div>
      <div className="sidebar-ws">
        <span className={`ws-dot ${live ? 'connected' : 'disconnected'}`} />
        {live ? 'WebSocket Connected' : 'Connecting...'}
      </div>

      {/* Next Prediction Card OR Cooling Down Card */}
      {cdActive ? (
        <div className="sidebar-prediction" style={{ borderColor: 'rgba(144,224,239,0.3)', background: 'rgba(144,224,239,0.1)' }}>
          <div className="sp-header">
            <span className="sp-title" style={{ color: '#90e0ef' }}>Cooling Down</span>
            <span className="sp-countdown" style={{ color: '#90e0ef', background: 'transparent' }}>{fmtRemain(cdTimer)}</span>
          </div>
          <div className="sp-signal">
            <div className="sp-signal-badge skip" style={{ color: '#90e0ef', borderColor: '#90e0ef' }}>❄️ PAUSED</div>
            <span className="sp-target">WAIT</span>
          </div>
          <div className="sp-meter" style={{ opacity: 0.5 }}>
            <div className="sp-meter-row">
              <span className="sp-meter-label">Confidence</span>
              <span className="sp-meter-val" style={{ color: '#90e0ef' }}>0.00%</span>
            </div>
            <div className="sp-bar-track">
              <div className="sp-bar-fill" style={{ width: `0%`, background: '#90e0ef' }} />
            </div>
          </div>
          <div className="sp-reason" style={{ color: '#90e0ef', marginTop: '8px' }}>
            Consecutive losses triggered safety pause. Trading will resume after timer ends.
          </div>
        </div>
      ) : (
        <div className="sidebar-prediction">
          <div className="sp-header">
            <span className="sp-title">Next Prediction</span>
            <span className="sp-countdown">{fmtCD(countdown)}</span>
          </div>

          <div className="sp-signal">
            <div className={`sp-signal-badge ${direction.toLowerCase()}`}>
              {direction === 'BUY' ? '▲' : direction === 'SELL' ? '▼' : '—'} {direction}
            </div>
            <span className="sp-target">{targetTime}</span>
          </div>

          <div className="sp-meter">
            <div className="sp-meter-row">
              <span className="sp-meter-label">Confidence</span>
              <span className="sp-meter-val" style={{ color: barColor }}>{confidencePct}%</span>
            </div>
            <div className="sp-bar-track">
              <div className="sp-bar-fill" style={{ width: `${Math.min(parseFloat(confidencePct), 100)}%`, background: barColor }} />
            </div>
          </div>

          {reason && <div className="sp-reason">{reason}</div>}
        </div>
      )}

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="sidebar-nav-label">Analytics</div>
        {navItems.map(item => (
          <div
            key={item.key}
            className={`sidebar-nav-item ${(activePage ?? 'overview') === item.key ? 'active' : ''}`}
            onClick={() => onPageChange?.(item.key)}
          >
            <span className="sidebar-nav-icon">{item.icon}</span> {item.label}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        Bitcoin Quant Prediction
        {lastBet?.timestamp && (
          <div className="sidebar-last-bet">
            Last: {new Date(lastBet.timestamp).toLocaleTimeString()}
            {' '}{lastBet.direction?.includes('UP') ? 'BUY' : 'SELL'}
            {' '}{lastBet.outcome ? 'Won' : 'Lost'}
          </div>
        )}
      </div>
    </aside>
  )
}
