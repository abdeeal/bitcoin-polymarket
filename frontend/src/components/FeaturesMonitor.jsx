import React, { useState, useEffect, useMemo } from 'react';
import './css/FeaturesMonitor.css';

const API = '/api';

const SOURCE_LABELS = {
  binance_rest: 'Binance REST',
  binance_ws: 'Binance WebSocket',
  coinglass: 'Coinglass',
  deribit: 'Deribit',
  polymarket: 'Polymarket',
  internal: 'Internal Engine',
};

// const CATEGORY_ICONS = {
//   'Price Action': '📊',
//   'Volume': '📈',
//   'Technical': '⚙️',
//   'Order Book': '📋',
//   'Microstructure': '🔬',
//   'Cross-Asset': '🔗',
//   'Sentiment': '💡',
//   'Calendar': '📅',
//   'Lagged & Interaction': '🧮',
// };

const CATEGORY_ICONS = {
  'Price Action': '',
  'Volume': '',
  'Technical': '',
  'Order Book': '',
  'Microstructure': '',
  'Cross-Asset': '',
  'Sentiment': '',
  'Calendar': '',
  'Lagged & Interaction': '',
};

export default function FeaturesMonitor() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [filterCategory, setFilterCategory] = useState('All');
  const [filterStatus, setFilterStatus] = useState('All');
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedSources, setExpandedSources] = useState({});

  const fetchHealth = async () => {
    try {
      const response = await fetch(`${API}/system/features-health`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Get unique categories
  const categories = useMemo(() => {
    if (!data?.features) return [];
    return ['All', ...new Set(data.features.map(f => f.category))];
  }, [data]);

  // Filtered features
  const filteredFeatures = useMemo(() => {
    if (!data?.features) return [];
    return data.features.filter(f => {
      if (filterCategory !== 'All' && f.category !== filterCategory) return false;
      if (filterStatus !== 'All' && f.status !== filterStatus) return false;
      if (searchTerm && !f.feature.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    });
  }, [data, filterCategory, filterStatus, searchTerm]);

  const toggleSource = (key) => {
    setExpandedSources(prev => ({ ...prev, [key]: !prev[key] }));
  };

  if (loading && !data) {
    return (
      <div className="fm-container">
        <div className="fm-loading">
          <div className="fm-spinner" />
          <span>Checking feature health...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="fm-container">
      {/* Header */}
      <div className="fm-header">
        <div className="fm-header-left">
          <h2 className="fm-title">Features Health Monitor</h2>
          <span className="fm-subtitle">
            {data?.total ?? 0} features across {Object.keys(data?.sources ?? {}).length} data sources
          </span>
        </div>
        <div className="fm-header-right">
          {lastUpdated && (
            <span className="fm-updated">
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button className="fm-refresh-btn" onClick={fetchHealth}>
            ↻ Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="fm-error-banner">
          <span>⚠ Failed to fetch: {error}</span>
        </div>
      )}

      {/* Summary Cards */}
      {data && (
        <div className="fm-summary-row">
          <div className="fm-summary-card fm-card-online">
            <span className="fm-sc-value">{data.online}</span>
            <span className="fm-sc-label">Online</span>
          </div>
          <div className="fm-summary-card fm-card-offline">
            <span className="fm-sc-value">{data.offline}</span>
            <span className="fm-sc-label">Offline</span>
          </div>
          <div className="fm-summary-card fm-card-notconfig">
            <span className="fm-sc-value">{data.not_configured}</span>
            <span className="fm-sc-label">Not Configured</span>
          </div>
          <div className="fm-summary-card fm-card-total">
            <span className="fm-sc-value">{data.total}</span>
            <span className="fm-sc-label">Total Features</span>
          </div>
        </div>
      )}

      {/* Data Sources Overview */}
      {data?.sources && (
        <div className="fm-sources-section">
          <div className="fm-section-title">Data Sources</div>
          <div className="fm-sources-grid">
            {Object.entries(data.sources).map(([key, src]) => (
              <div
                key={key}
                className={`fm-source-card ${src.status === 'Online' ? 'online' : src.status === 'Not Configured' ? 'notconfig' : 'offline'}`}
                onClick={() => toggleSource(key)}
              >
                <div className="fm-src-header">
                  <span className={`fm-src-dot ${src.status === 'Online' ? 'on' : src.status === 'Not Configured' ? 'nc' : 'off'}`} />
                  <span className="fm-src-name">{SOURCE_LABELS[key] || key}</span>
                  <span className={`fm-src-badge ${src.status === 'Online' ? 'badge-on' : src.status === 'Not Configured' ? 'badge-nc' : 'badge-off'}`}>
                    {src.status}
                  </span>
                </div>
                <div className="fm-src-note">{src.note}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="fm-filters">
        <div className="fm-section-title">Feature Details</div>
        <div className="fm-filter-row">
          <div className="fm-filter-group">
            <label className="fm-filter-label">Category</label>
            <select
              className="fm-select"
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
            >
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="fm-filter-group">
            <label className="fm-filter-label">Status</label>
            <select
              className="fm-select"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
            >
              <option value="All">All</option>
              <option value="Online">Online</option>
              <option value="Offline">Offline</option>
              <option value="Not Configured">Not Configured</option>
            </select>
          </div>
          <div className="fm-filter-group fm-search-group">
            <label className="fm-filter-label">Search</label>
            <input
              className="fm-search"
              type="text"
              placeholder="Filter features..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <span className="fm-filter-count">
            {filteredFeatures.length} / {data?.total ?? 0}
          </span>
        </div>
      </div>

      {/* Features Table */}
      <div className="fm-table-container">
        <table className="fm-table">
          <thead>
            <tr>
              <th className="fm-th-num">#</th>
              <th className="fm-th-feature">Feature</th>
              <th className="fm-th-category">Category</th>
              <th className="fm-th-source">Data Source</th>
              <th className="fm-th-status">Status</th>
              <th className="fm-th-note">Note</th>
            </tr>
          </thead>
          <tbody>
            {filteredFeatures.map((f, i) => (
              <tr key={f.id} className={`fm-row fm-row-${f.status === 'Online' ? 'on' : f.status === 'Not Configured' ? 'nc' : 'off'}`}>
                <td className="fm-td-num">{f.id}</td>
                <td className="fm-td-feature">
                  <code>{f.feature}</code>
                </td>
                <td className="fm-td-category">
                  <span className="fm-cat-badge">
                    {CATEGORY_ICONS[f.category] || ''} {f.category}
                  </span>
                </td>
                <td className="fm-td-source">
                  <span className="fm-api-url">{f.api_url}</span>
                </td>
                <td className="fm-td-status">
                  <span className={`fm-status-pill ${f.status === 'Online' ? 'pill-on' : f.status === 'Not Configured' ? 'pill-nc' : 'pill-off'}`}>
                    {f.status}
                  </span>
                </td>
                <td className={`fm-td-note ${f.status === 'Online' ? 'note-on' : 'note-err'}`}>
                  {f.note}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
