/** Panel 6 — Trade Statistics */
export default function Panel6_Streaks({ streaks }) {
  if (!streaks) return <div className="panel-loading">Loading...</div>
  const wr = ((streaks.win_rate ?? 0) * 100).toFixed(1)
  const pf = streaks.profit_factor?.toFixed(2) ?? '0.00'

  return (
    <div>
      <div className="panel-title" style={{ marginBottom: 12 }}>Trade Statistics</div>
      <div className="stats-grid">
        <div className="stat-item"><span className="stat-label">Total Entries</span><span className="stat-val">{streaks.total_entries ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Wins</span><span className="stat-val green">{streaks.total_wins ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Losses</span><span className="stat-val red">{streaks.total_losses ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Win Rate</span><span className="stat-val accent">{wr}%</span></div>
        <div className="stat-item"><span className="stat-label">Max Win Streak</span><span className="stat-val green">{streaks.max_win_streak ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Max Loss Streak</span><span className="stat-val red">{streaks.max_lose_streak ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Avg Win</span><span className="stat-val green">+${streaks.avg_win_usd?.toFixed(2) ?? '0'}</span></div>
        <div className="stat-item"><span className="stat-label">Avg Loss</span><span className="stat-val red">${streaks.avg_loss_usd?.toFixed(2) ?? '0'}</span></div>
        <div className="stat-item"><span className="stat-label">Profit Factor</span><span className="stat-val accent">{pf}</span></div>
        <div className="stat-item"><span className="stat-label">Largest Win</span><span className="stat-val green">+${streaks.largest_win_usd?.toFixed(2) ?? '0'}</span></div>
        <div className="stat-item"><span className="stat-label">Curr Win Streak</span><span className="stat-val green">{streaks.current_win_streak ?? 0}</span></div>
        <div className="stat-item"><span className="stat-label">Curr Loss Streak</span><span className="stat-val red">{streaks.current_lose_streak ?? 0}</span></div>
      </div>
    </div>
  )
}
