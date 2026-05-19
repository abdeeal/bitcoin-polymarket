/**
 * TradeHistory — Full paginated trade log with date filter and detail expansion.
 *
 * Features:
 *  - Pagination: All / 20 / 50 / 100 rows per page
 *  - Date range filter (start date → end date)
 *  - Columns: No, Direction, Confidence, Result, PnL, Balance, Detail
 *  - Detail panel: per-model confidence bars + feature pro/contra
 */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'

const API = '/api'
const fetcher = (url) => fetch(API + url).then(r => r.json())

const PAGE_SIZES = [
  { label: 'All', value: 0 },
  { label: '20',  value: 20 },
  { label: '50',  value: 50 },
  { label: '100', value: 100 },
]

function fmt(date) {
  if (!date) return ''
  return new Date(date).toLocaleString('id-ID', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  })
}

function DetailPanel({ betId, onClose }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['trade-detail', betId],
    queryFn: () => fetcher(`/trades/detail/${betId}`),
    enabled: !!betId,
    staleTime: 60_000,
  })

  if (isLoading) {
    return (
      <tr className="td-detail-row">
        <td colSpan="8">
          <div className="td-detail-panel">
            <div className="td-detail-loading">Loading model details...</div>
          </div>
        </td>
      </tr>
    )
  }

  if (error || data?.error) {
    return (
      <tr className="td-detail-row">
        <td colSpan="8">
          <div className="td-detail-panel">
            <div className="td-detail-empty">
              {data?.error || 'No detail data available for this trade. Details are recorded from new trades onward.'}
            </div>
          </div>
        </td>
      </tr>
    )
  }

  const models = [
    { key: 'lgbm', label: 'LightGBM', weight: '50%', prob: data.prob_lgbm },
    { key: 'xgb', label: 'XGBoost', weight: '15%', prob: data.prob_xgb },
    { key: 'lstm', label: 'LSTM', weight: '35%', prob: data.prob_lstm },
  ]

  const isUp = data.signal?.includes('UP')
  const ensembleConf = isUp ? data.prob_ensemble : 1 - data.prob_ensemble

  return (
    <tr className="td-detail-row">
      <td colSpan="8">
        <div className="td-detail-panel">
          {/* Header */}
          <div className="td-dp-header">
            <span className="td-dp-title">Model Breakdown & Feature Attribution</span>
            <button className="td-dp-close" onClick={onClose}>✕</button>
          </div>

          {/* Model Confidence Bars */}
          <div className="td-dp-models">
            <div className="td-dp-section-title">Per-Model Confidence</div>
            <div className="td-dp-model-grid">
              {/* Ensemble */}
              <div className="td-dp-model-card td-dp-ensemble">
                <div className="td-dp-model-header">
                  <span className="td-dp-model-name">Ensemble</span>
                  <span className="td-dp-model-weight">Combined</span>
                </div>
                <div className="td-dp-model-bar-wrap">
                  <div
                    className="td-dp-model-bar"
                    style={{
                      width: `${(ensembleConf * 100).toFixed(0)}%`,
                      background: 'linear-gradient(90deg, var(--accent), var(--blue))'
                    }}
                  />
                </div>
                <span className="td-dp-model-prob">{(ensembleConf * 100).toFixed(2)}%</span>
              </div>

              {models.map(m => {
                const p = m.prob
                const conf = p != null ? (isUp ? p : 1 - p) : null
                const available = conf != null
                return (
                  <div key={m.key} className={`td-dp-model-card ${!available ? 'disabled' : ''}`}>
                    <div className="td-dp-model-header">
                      <span className="td-dp-model-name">{m.label}</span>
                      <span className="td-dp-model-weight">w: {m.weight}</span>
                    </div>
                    {available ? (
                      <>
                        <div className="td-dp-model-bar-wrap">
                          <div
                            className="td-dp-model-bar"
                            style={{
                              width: `${(conf * 100).toFixed(0)}%`,
                              background: conf >= 0.7 ? 'var(--green)' : conf >= 0.55 ? 'var(--yellow)' : 'var(--muted)'
                            }}
                          />
                        </div>
                        <span className="td-dp-model-prob">{(conf * 100).toFixed(2)}%</span>
                      </>
                    ) : (
                      <div className="td-dp-model-na">Not Available</div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* Feature Pro/Contra */}
          <div className="td-dp-features">
            <div className="td-dp-features-cols">
              {/* PRO */}
              <div className="td-dp-feat-col">
                <div className="td-dp-feat-title pro">▲ Supporting Features (Pro)</div>
                {data.features_pro?.length > 0 ? (
                  <table className="td-dp-feat-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.features_pro.map((f, i) => (
                        <tr key={i}>
                          <td><code>{f.name}</code></td>
                          <td className="td-dp-feat-imp">{f.importance.toFixed(1)}</td>
                          <td className="td-dp-feat-val">{f.value.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="td-dp-feat-empty">No data</div>
                )}
              </div>

              {/* CONTRA */}
              <div className="td-dp-feat-col">
                <div className="td-dp-feat-title contra">▼ Opposing Features (Contra)</div>
                {data.features_contra?.length > 0 ? (
                  <table className="td-dp-feat-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.features_contra.map((f, i) => (
                        <tr key={i}>
                          <td><code>{f.name}</code></td>
                          <td className="td-dp-feat-imp">{f.importance.toFixed(1)}</td>
                          <td className="td-dp-feat-val">{f.value.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="td-dp-feat-empty">No data</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </td>
    </tr>
  )
}

export default function TradeHistory() {
  const [pageSize, setPageSize] = useState(20)
  const [page, setPage]         = useState(1)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo,   setDateTo]   = useState('')
  const [expandedId, setExpandedId] = useState(null)

  const { data: trades = [], isLoading } = useQuery({
    queryKey: ['trade-history'],
    queryFn: () => fetcher('/trades/history'),
    staleTime: 15_000,
    refetchInterval: 15_000,
  })

  // Apply date filter
  const filtered = useMemo(() => {
    return trades.filter(t => {
      const ts = new Date(t.timestamp)
      if (dateFrom && ts < new Date(dateFrom)) return false
      if (dateTo) {
        const end = new Date(dateTo)
        end.setHours(23, 59, 59, 999)
        if (ts > end) return false
      }
      return true
    })
  }, [trades, dateFrom, dateTo])

  // Paginate
  const totalPages = pageSize === 0 ? 1 : Math.max(1, Math.ceil(filtered.length / pageSize))
  const safePage   = Math.min(page, totalPages)
  const paginated  = pageSize === 0
    ? filtered
    : filtered.slice((safePage - 1) * pageSize, safePage * pageSize)

  const handlePageSizeChange = (v) => { setPageSize(v); setPage(1) }
  const handleDateFrom = (v) => { setDateFrom(v); setPage(1) }
  const handleDateTo   = (v) => { setDateTo(v);   setPage(1) }
  const clearFilter    = () => { setDateFrom(''); setDateTo(''); setPage(1) }

  // Summary stats
  const totalPnl   = filtered.reduce((s, t) => s + (t.pnl ?? 0), 0)
  const winCount   = filtered.filter(t => t.outcome === true).length
  const winRate    = filtered.length > 0 ? (winCount / filtered.length * 100).toFixed(1) : '0.0'

  const toggleDetail = (id) => {
    setExpandedId(prev => prev === id ? null : id)
  }

  return (
    <div className="trade-history-container">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="th-header">
        <div className="th-title-row">
          <h2 className="th-title">Trade History</h2>
          <div className="th-summary">
            <span className="th-badge">{filtered.length} trades</span>
            <span className={`th-badge ${parseFloat(totalPnl) >= 0 ? 'th-badge-green' : 'th-badge-red'}`}>
              PnL: {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
            </span>
            <span className="th-badge">WR: {winRate}%</span>
          </div>
        </div>

        {/* ── Controls ──────────────────────────────────────────── */}
        <div className="th-controls">
          {/* Date Range */}
          <div className="th-filter-group">
            <label className="th-filter-label">From</label>
            <input
              type="date"
              className="th-date-input"
              value={dateFrom}
              onChange={e => handleDateFrom(e.target.value)}
            />
          </div>
          <div className="th-filter-group">
            <label className="th-filter-label">To</label>
            <input
              type="date"
              className="th-date-input"
              value={dateTo}
              onChange={e => handleDateTo(e.target.value)}
            />
          </div>
          {(dateFrom || dateTo) && (
            <button className="th-clear-btn" onClick={clearFilter}>✕ Clear</button>
          )}

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Rows per page */}
          <div className="th-filter-group">
            <label className="th-filter-label">Rows</label>
            <div className="toggle-group">
              {PAGE_SIZES.map(s => (
                <button
                  key={s.value}
                  className={pageSize === s.value ? 'active' : ''}
                  onClick={() => handlePageSizeChange(s.value)}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Table ─────────────────────────────────────────────── */}
      <div className="th-table-wrap">
        {isLoading ? (
          <div className="th-empty">Loading trade history…</div>
        ) : filtered.length === 0 ? (
          <div className="th-empty">No trades found{(dateFrom || dateTo) ? ' for selected date range' : ''}.</div>
        ) : (
          <table className="th-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th>Direction</th>
                <th>Confidence</th>
                <th>Result</th>
                <th>PnL</th>
                <th>Balance</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {paginated.map((t, i) => {
                const rowNum = pageSize === 0
                  ? i + 1
                  : (safePage - 1) * pageSize + i + 1
                const isUp    = t.signal?.includes('UP')
                const conf    = typeof t.model_prob === 'number'
                  ? (isUp ? t.model_prob : 1 - t.model_prob) * 100
                  : 0
                const pnlPos  = (t.pnl ?? 0) >= 0
                const isExpanded = expandedId === t.id

                return (
                  <>
                    <tr key={t.id ?? i} className={t.outcome ? 'tr-win' : 'tr-loss'}>
                      <td className="td-num">{rowNum}</td>
                      <td className="td-time">{fmt(t.timestamp)}</td>
                      <td>
                        <span className={`trade-badge ${isUp ? 'buy' : 'sell'}`}>
                          {isUp ? '▲ BUY' : '▼ SELL'}
                        </span>
                        <span className={`trade-badge ${isUp ? 'buy' : 'sell'}`} style={{ backgroundColor: 'transparent',fontWeight: 'bold', marginLeft: '12px', }}>{conf.toFixed(2) >= 82.00 ? '⚡︎' : ''}</span>
                      </td>
                      <td className="td-conf">
                        <div className="conf-wrap">
                          <span className="conf-val">{conf.toFixed(2)}%</span>
                          <div className="conf-bar-track">
                            <div
                              className="conf-bar-fill"
                              style={{
                                width: `${Math.min(conf, 100)}%`,
                                background: conf >= 70 ? 'var(--green)' : conf >= 55 ? 'var(--yellow)' : 'var(--muted)'
                              }}
                            />
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className={`trade-outcome ${t.outcome ? 'win' : 'loss'}`}>
                          {t.outcome ? 'WIN' : 'LOSS'}
                        </span>
                      </td>
                      <td className={pnlPos ? 'green' : 'red'}>
                        {pnlPos ? '+' : ''}{(t.pnl ?? 0).toFixed(2)}
                      </td>
                      <td className="td-bal">${(t.balance_after ?? 0).toFixed(2)}</td>
                      <td>
                        <button
                          className={`td-detail-btn ${isExpanded ? 'active' : ''}`}
                          onClick={() => toggleDetail(t.id)}
                          title="View model breakdown"
                        >
                          {isExpanded ? '▾' : '▸'} Detail
                        </button>
                      </td>
                    </tr>
                    {isExpanded && (
                      <DetailPanel
                        key={`detail-${t.id}`}
                        betId={t.id}
                        onClose={() => setExpandedId(null)}
                      />
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Pagination Controls ────────────────────────────────── */}
      {pageSize !== 0 && totalPages > 1 && (
        <div className="th-pagination">
          <button
            className="th-pg-btn"
            disabled={safePage === 1}
            onClick={() => setPage(1)}
          >«</button>
          <button
            className="th-pg-btn"
            disabled={safePage === 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
          >‹</button>

          {/* Page numbers (show window of 5) */}
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let start = Math.max(1, safePage - 2)
            if (start + 4 > totalPages) start = Math.max(1, totalPages - 4)
            return start + i
          }).map(p => (
            <button
              key={p}
              className={`th-pg-btn ${p === safePage ? 'th-pg-active' : ''}`}
              onClick={() => setPage(p)}
            >{p}</button>
          ))}

          <button
            className="th-pg-btn"
            disabled={safePage === totalPages}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
          >›</button>
          <button
            className="th-pg-btn"
            disabled={safePage === totalPages}
            onClick={() => setPage(totalPages)}
          >»</button>

          <span className="th-pg-info">
            Page {safePage} of {totalPages} · {filtered.length} total
          </span>
        </div>
      )}
    </div>
  )
}
