-- =============================================================================
-- BTC Prediction Bot — TimescaleDB Schema
-- =============================================================================
-- This schema is auto-applied on first docker-compose boot via the init script.
-- All tables use TimescaleDB hypertables for time-series optimization.
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- 1. raw_candles_1m — 1-minute OHLCV candles from Binance
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_candles_1m (
    timestamp        TIMESTAMPTZ NOT NULL,
    open             NUMERIC(12,2),
    high             NUMERIC(12,2),
    low              NUMERIC(12,2),
    close            NUMERIC(12,2),
    volume           NUMERIC(20,8),
    taker_buy_volume NUMERIC(20,8),
    trade_count      INTEGER,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('raw_candles_1m', 'timestamp', if_not_exists => TRUE);

-- Index for fast range queries during backfill verification
CREATE INDEX IF NOT EXISTS idx_candles_1m_timestamp
    ON raw_candles_1m (timestamp DESC);

-- =============================================================================
-- 2. raw_orderbook_snapshots — L2 order book snapshots
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_orderbook_snapshots (
    timestamp        TIMESTAMPTZ NOT NULL,
    bid_depth        NUMERIC(20,8),
    ask_depth        NUMERIC(20,8),
    spread           NUMERIC(12,8),
    mid_price        NUMERIC(12,2),
    bid_depth_1pct   NUMERIC(20,8),
    ask_depth_1pct   NUMERIC(20,8),
    bid_ask_ratio_5  NUMERIC(10,6),
    bid_ask_ratio_10 NUMERIC(10,6),
    raw_bids         JSONB,
    raw_asks         JSONB,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('raw_orderbook_snapshots', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- 3. raw_trades — Aggregated trade events (sampled, not every tick)
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_trades (
    timestamp        TIMESTAMPTZ NOT NULL,
    price            NUMERIC(12,2),
    quantity         NUMERIC(20,8),
    is_buyer_maker   BOOLEAN,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('raw_trades', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- 4. raw_funding_oi — Funding rate and open interest snapshots
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_funding_oi (
    timestamp        TIMESTAMPTZ NOT NULL,
    funding_rate     NUMERIC(12,8),
    open_interest    NUMERIC(20,8),
    mark_price       NUMERIC(12,2),
    index_price      NUMERIC(12,2),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('raw_funding_oi', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- 5. raw_liquidations — Liquidation events from Coinglass
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_liquidations (
    timestamp        TIMESTAMPTZ NOT NULL,
    long_liq_usd     NUMERIC(20,2),
    short_liq_usd    NUMERIC(20,2),
    liq_ratio        NUMERIC(10,6),
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('raw_liquidations', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- 6. features_5m — Computed feature vectors (written async by persistence worker)
-- =============================================================================
CREATE TABLE IF NOT EXISTS features_5m (
    timestamp  TIMESTAMPTZ NOT NULL,
    label      SMALLINT,               -- 1=up, 0=down, NULL=pending
    bot_state  TEXT,                    -- ACTIVE | COOLING_DOWN | BOOT
    features   JSONB NOT NULL,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('features_5m', 'timestamp', if_not_exists => TRUE);

-- Index for training data queries
CREATE INDEX IF NOT EXISTS idx_features_5m_label
    ON features_5m (timestamp DESC)
    WHERE label IS NOT NULL;

-- =============================================================================
-- 7. model_predictions — Raw model output per cycle
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_predictions (
    timestamp        TIMESTAMPTZ NOT NULL,
    model_prob       NUMERIC(6,4),
    signal           TEXT,
    confidence_tier  TEXT,
    bot_state        TEXT,
    executed         BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (timestamp)
);

SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);

-- =============================================================================
-- 8. bet_log — Full bet history with feature snapshots
-- =============================================================================
CREATE TABLE IF NOT EXISTS bet_log (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    market_id        TEXT NOT NULL,
    signal           TEXT NOT NULL,          -- BET_UP_NORMAL | BET_UP_HIGH_CONF | etc.
    confidence_tier  TEXT NOT NULL,          -- NORMAL | HIGH_CONF
    model_prob       NUMERIC(6,4),
    bet_size         NUMERIC(10,2),
    odds             NUMERIC(6,4),
    outcome          BOOLEAN,
    pnl              NUMERIC(10,2),
    balance_before   NUMERIC(10,2),
    balance_after    NUMERIC(10,2),
    bracket_index    INTEGER,
    paper_mode       BOOLEAN DEFAULT TRUE,
    feature_snapshot JSONB
);

-- Create hypertable on bet_log — needs timestamp for partitioning
-- Since id is UUID, we add timestamp as a dimension
SELECT create_hypertable('bet_log', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_bet_log_timestamp
    ON bet_log (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_bet_log_outcome
    ON bet_log (outcome)
    WHERE outcome IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_bet_log_signal
    ON bet_log (signal);

-- =============================================================================
-- 8b. bet_details — Per-model confidence and feature attribution
-- =============================================================================
CREATE TABLE IF NOT EXISTS bet_details (
    bet_id           UUID NOT NULL REFERENCES bet_log(id) ON DELETE CASCADE,
    prob_lgbm        NUMERIC(6,4),
    prob_xgb         NUMERIC(6,4),
    prob_lstm        NUMERIC(6,4),
    prob_ensemble    NUMERIC(6,4),
    features_pro     JSONB,            -- top 10 features supporting the direction
    features_contra  JSONB,            -- top 10 features opposing the direction
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (bet_id)
);

-- =============================================================================
-- 9. cooling_down_log — Cooling down event history
-- =============================================================================
CREATE TABLE IF NOT EXISTS cooling_down_log (
    id               SERIAL PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ,
    trigger_reason   TEXT NOT NULL DEFAULT '2_consecutive_losses',
    duration_sec     INTEGER,
    balance_at_start NUMERIC(10,2),
    bracket_at_start INTEGER,
    candles_missed   INTEGER DEFAULT 0      -- should always be 0
);

-- =============================================================================
-- 10. bot_session_log — Bot uptime tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS bot_session_log (
    id         SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at   TIMESTAMPTZ,
    boot_type  TEXT,       -- 'cold' | 'warm'
    reason     TEXT        -- 'normal_shutdown' | 'crash' | 'hard_stop'
);

-- =============================================================================
-- 11. portfolio_state — Current portfolio snapshot (single row, upserted)
-- =============================================================================
CREATE TABLE IF NOT EXISTS portfolio_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    balance         NUMERIC(10,2) NOT NULL DEFAULT 10.00,
    bracket_index   INTEGER NOT NULL DEFAULT 0,
    total_bets      INTEGER NOT NULL DEFAULT 0,
    total_wins      INTEGER NOT NULL DEFAULT 0,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (id = 1)  -- ensure single row
);

-- Insert initial state if not exists
INSERT INTO portfolio_state (id, balance)
VALUES (1, 10.00)
ON CONFLICT (id) DO NOTHING;

-- =============================================================================
-- Data Retention Policies (optional — uncomment if storage is a concern)
-- =============================================================================
-- Raw data older than 90 days can be compressed/dropped
-- SELECT add_retention_policy('raw_orderbook_snapshots', INTERVAL '30 days');
-- SELECT add_retention_policy('raw_trades', INTERVAL '7 days');

-- Compression policy for older data
-- ALTER TABLE raw_candles_1m SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = ''
-- );
-- SELECT add_compression_policy('raw_candles_1m', INTERVAL '7 days');
