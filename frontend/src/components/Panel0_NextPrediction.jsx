/** Panel 0 — Next Prediction Card */
import { useState, useEffect } from 'react'

export default function Panel0_NextPrediction({ live, predictionData }) {
  // Prefer WebSocket live data, fall back to REST query
  const pred = live?.next_prediction ?? predictionData ?? null
  const [countdown, setCountdown] = useState(0)

  useEffect(() => {
    if (!pred) {
      // Compute local countdown to next 5-min candle
      const now = Math.floor(Date.now() / 1000)
      const interval = 300
      const nextTs = Math.ceil(now / interval) * interval
      setCountdown(Math.max(0, nextTs - now))
    } else {
      setCountdown(pred.remaining_seconds ?? 0)
    }
    const id = setInterval(() => {
      setCountdown(prev => (prev > 0 ? prev - 1 : 0))
    }, 1000)
    return () => clearInterval(id)
  }, [pred?.target_time])

  // Re-sync countdown when it hits 0 (new 5-min window)
  useEffect(() => {
    if (countdown === 0) {
      const timer = setTimeout(() => {
        const now = Math.floor(Date.now() / 1000)
        const interval = 300
        const nextTs = Math.ceil(now / interval) * interval
        setCountdown(Math.max(0, nextTs - now))
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [countdown])

  const direction = pred?.direction ?? 'SKIP'
  const confidence = pred?.confidence ?? 0
  const reason = pred?.reason ?? 'Waiting for data'

  // Compute target time locally if not provided
  const getTargetTime = () => {
    if (pred?.target_time) return pred.target_time
    const now = Math.floor(Date.now() / 1000)
    const nextTs = Math.ceil(now / 300) * 300
    const d = new Date(nextTs * 1000)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  const targetTime = getTargetTime()

  const dirConfig = {
    BUY:  { label: 'BUY',  icon: '▲', color: '#22c55e', bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.3)' },
    SELL: { label: 'SELL', icon: '▼', color: '#ef4444', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)' },
    SKIP: { label: 'SKIP', icon: '—', color: '#94a3b8', bg: 'rgba(148,163,184,0.08)', border: 'rgba(148,163,184,0.2)' },
  }

  const cfg = dirConfig[direction] ?? dirConfig.SKIP
  const confidencePct = Math.round(confidence * 100)
  const rawProb = pred?.raw_prob ?? confidence
  const rawProbPct = Math.round(rawProb * 100)
  const barColor = confidencePct >= 70 ? '#22c55e' : confidencePct >= 55 ? '#f59e0b' : '#94a3b8'

  const fmtCountdown = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${String(sec).padStart(2, '0')}`
  }

  return (
    <div className="panel panel-prediction" style={{ borderColor: cfg.border }}>
      <div className="pred-header">
        <h3 className="panel-title">Next Prediction</h3>
        <div className="pred-countdown" title="Time until next 5-min candle">
          <span className="pred-countdown-icon">&#9201;</span>
          <span className="pred-countdown-val">{fmtCountdown(countdown)}</span>
        </div>
      </div>

      <div className="pred-body">
        <div className="pred-signal" style={{ background: cfg.bg, borderColor: cfg.border }}>
          <span className="pred-icon" style={{ color: cfg.color }}>{cfg.icon}</span>
          <span className="pred-direction" style={{ color: cfg.color }}>{cfg.label}</span>
        </div>

        <div className="pred-meta">
          <div className="pred-target-row">
            <span className="pred-label">Target</span>
            <span className="pred-target-time">{targetTime}</span>
          </div>

          <div className="pred-conf-row">
            <span className="pred-label">Confidence</span>
            <span className="pred-conf-val" style={{ color: barColor }}>
              {confidencePct}%
            </span>
          </div>
          <div className="pred-conf-bar-wrap">
            <div
              className="pred-conf-bar"
              style={{
                width: `${confidencePct}%`,
                background: `linear-gradient(90deg, ${barColor}, ${barColor}88)`,
              }}
            />
          </div>
          <div className="pred-prob-row">
            <span className="pred-label">P(UP)</span>
            <span className="pred-prob-val">{rawProbPct}%</span>
          </div>
          {reason && <div className="pred-reason">{reason}</div>}
        </div>
      </div>
    </div>
  )
}
