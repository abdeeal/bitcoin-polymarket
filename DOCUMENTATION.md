# DOCUMENTATION.md — BTC 5-Minute Prediction Bot (Opus 1.0)

> **Purpose**: This file provides comprehensive project context for developers and AI agents
> so they can immediately understand the architecture, data flow, and every module without
> reading all source files.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Module Reference](#3-module-reference)
4. [Data Flow](#4-data-flow)
5. [State Machine](#5-state-machine)
6. [Compounding & Bet Sizing](#6-compounding--bet-sizing)
7. [Feature Engine](#7-feature-engine)
8. [Signal Generation](#8-signal-generation)
9. [Execution Engine](#9-execution-engine)
10. [Dashboard](#10-dashboard)
11. [Monitoring & Alerting](#11-monitoring--alerting)
12. [Testing Strategy](#12-testing-strategy)
13. [Production Debugging Notes](#13-production-debugging-notes)
14. [Configuration & Environment](#14-configuration--environment)
15. [Deployment](#15-deployment)

---

## 1. Project Overview

An AI-powered binary options prediction system targeting **Polymarket's BTC 5-minute markets**.
The bot continuously:
1. Ingests real-time BTC candle data (Binance WebSocket)
2. Computes 150+ features in <100ms (NumPy/Numba, zero allocation)
3. Runs ensemble ML model (LightGBM + XGBoost + LSTM)
4. Generates BET_UP or BET_DOWN signals when confidence exceeds thresholds
5. Sizes bets using a fixed bracket system ($5–$99) or fractional Kelly ($100+)
6. Executes on Polymarket CLOB via async REST API (paper or live mode)
7. Monitors performance, detects feature drift, and auto-retrains weekly

**Key Numbers**:
- Starting balance: $10 USD
- Signal thresholds: P ≥ 0.72 (normal), P ≥ 0.82 (high confidence)
- Cooling down: 30 minutes after 2 consecutive losses
- Feature latency budget: P95 < 100ms
- Candle interval: 5 minutes (300 seconds)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        I/O Layer (asyncio)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Binance WS  │  │ REST Backfill│  │ Derivatives/Cross-Asset   │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────────┘ │
│         │                │                      │                  │
│         └────────────────┼──────────────────────┘                  │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │   Ring Buffer (600)    │  ◄── Pre-allocated numpy   │
│              │   30 fields × float64  │      Zero runtime alloc    │
│              └───────────┬───────────┘                             │
│                          │                                          │
├──────────────────────────┼──────────────────────────────────────────┤
│                   CPU Layer (thread pool)                           │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │   Feature Engine       │  ◄── 150+ features <100ms  │
│              │   NumPy + Numba JIT    │                             │
│              └───────────┬───────────┘                             │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │   Normalizer (z-score) │                             │
│              └───────────┬───────────┘                             │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │   Ensemble Model       │  ◄── LightGBM+XGB+LSTM    │
│              └───────────┬───────────┘                             │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │  Signal Generator      │  ◄── Two-tier thresholds   │
│              └───────────┬───────────┘                             │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │  Risk Gates (6 checks) │                             │
│              └───────────┬───────────┘                             │
│                          ▼                                          │
├──────────────────────────┼──────────────────────────────────────────┤
│                  Execution Layer                                    │
│                          ▼                                          │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ Compounding│  │ Market Sel.  │  │ Polymarket Client           │ │
│  │ Engine     │  │ (best odds)  │  │ (retry + circuit breaker)   │ │
│  └─────┬─────┘  └──────┬───────┘  └──────────┬──────────────────┘ │
│        │               │                      │                    │
│        └───────────────┼──────────────────────┘                    │
│                        ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Paper Trader (validation) / Live Execution                   │  │
│  └──────────────────────────────────────┬───────────────────────┘  │
│                                         ▼                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Loss Tracker │  │ Bet Logger   │  │ Persistence Queue → DB   │  │
│  └─────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Critical Rules
- **I/O layer NEVER runs numpy** — it only writes raw dicts to Ring Buffer
- **CPU layer NEVER touches network** — it only reads numpy arrays from Ring Buffer
- **DB writes are fire-and-forget** — pushed to async queue, never block the hot path
- **Features compute ALWAYS** — even during COOLING_DOWN (no data gaps)

---

## 3. Module Reference

### `src/memory/ring_buffer.py`
- **RingBuffer**: Pre-allocated circular buffer, 600 bars × 30 fields
- Thread-safe: RLock for writes, numpy slicing for reads
- `push(bar)` / `push_batch(bars)` / `get_window(n)` / `get_field_window(field, n)`
- Edge cases: wrap-around concatenation, empty buffer → empty arrays

### `src/ingestion/`
- **binance_ws.py**: WebSocket → Ring Buffer (I/O only)
- **binance_rest.py**: REST backfill for buffer initialization
- **derivatives_client.py**: Funding rate, OI, liquidations
- **data_manager.py**: Orchestrates all data sources

### `src/features/`
- **feature_engine.py**: Async orchestrator with CPU thread pool
- **price_action.py**: 20 features (returns, candle ratios, consecutive patterns)
- **volume.py**: 15 features (VWAP, volume ratios, delta)
- **technical.py**: 35 features (RSI, MACD, BB, EMA, ATR, Stochastic)
- **order_book.py**: 20 features (depth imbalance, spread, book pressure)
- **microstructure.py**: 15 features (OFI, Kyle's lambda, trade intensity)
- **cross_asset.py**: 12 features (ETH/SOL correlation, funding)
- **sentiment.py**: 10 features (liquidations, OI change, put/call)
- **calendar.py**: 8 features (hour, day, session overlap)
- **lagged_interaction.py**: 15 features (lag + cross-feature products)
- **normalizer.py**: Rolling z-score/min-max normalization
- **jit_kernels.py**: Numba JIT-compiled hot-path functions

### `src/models/`
- **lgbm_model.py**: LightGBM gradient boosted trees
- **xgb_model.py**: XGBoost alternative
- **lstm_model.py**: PyTorch LSTM for sequence patterns
- **ensemble.py**: Weighted average of all models
- **calibration.py**: Probability calibration (Platt scaling)

### `src/signal/`
- **signal_generator.py**: Two-tier threshold classifier
  - `Signal` enum: NO_BET, BET_UP_NORMAL, BET_DOWN_NORMAL, BET_UP_HIGH_CONF, BET_DOWN_HIGH_CONF
  - `BotState` enum: BOOT, ACTIVE, COOLING_DOWN, PAUSED
- **risk_gates.py**: 6 pre-signal checks (spread, volume, time, etc.)
- **odds_checker.py**: EV computation against Polymarket odds

### `src/sizing/`
- **compounding_engine.py**: Fixed bracket system (6 brackets, $5–$100+)
  - Upgrade: balance ≥ ceiling AND ≥10 bets AND ≥50% win rate
  - Downgrade: balance < floor → immediate drop
  - No hard stop: betting continues until balance < $1.00
- **kelly_sizer.py**: Fractional Kelly (25%) with 2% cap for $100+
- **portfolio_tracker.py**: Balance and position tracking

### `src/execution/`
- **execution_engine.py**: 12-step main loop (PRD 12.2)
- **polymarket_client.py**: Async REST with 3× retry + 5-failure circuit breaker
- **market_selector.py**: Best market by volume + odds filter
- **paper_trader.py**: Simulated execution for validation
- **bet_logger.py**: Two-phase logging (placed → resolved)
- **loss_tracker.py**: Consecutive loss counter → cooling down trigger
- **state_manager.py**: BotState FSM with timer

### `src/backtest/`
- **backtester.py**: Bar-by-bar simulation mirroring production
- **cost_model.py**: Polymarket fee (2% on profit) + 5bps slippage
- **report_generator.py**: Streak analysis, PnL-by-signal, Sharpe
- **leakage_detector.py**: 4-check data leakage audit

### `src/dashboard/`
- **api.py**: FastAPI app with 9 REST endpoints + WebSocket
- **queries.py**: All 11 panel async SQL queries (asyncpg)
- **ws_broadcaster.py**: 5-second push to all WebSocket clients

### `src/monitoring/`
- **alerting.py**: AlertManager + 10 alert rules + Telegram/Email/Log channels
- **drift_detector.py**: PSI-based feature drift detection
- **performance_tracker.py**: Rolling latency, win rate, API error metrics

### `src/training/`
- **retrain_scheduler.py**: Weekly retrain pipeline (Sunday 00:00 UTC)
- **train_pipeline.py**: Full training workflow
- **walk_forward.py**: Walk-forward cross-validation
- **hyperopt_search.py**: Optuna hyperparameter optimization

---

## 4. Data Flow

```
Binance WS → push(bar) → Ring Buffer [600 bars]
                              ↓
                     get_window(200) → Feature Engine
                              ↓
                     150+ features → Normalizer → z-scored vector
                              ↓
                     Ensemble Model → P(UP) ∈ [0, 1]
                              ↓
                     Signal Generator → Signal enum
                              ↓
                     Risk Gates → filters_ok: bool
                              ↓
                     Compounding → bet_size: float
                              ↓
                     Paper/Live Execution → outcome: bool
                              ↓
                     Loss Tracker → cooling_down check
                              ↓
                     Bet Logger → persistence queue → TimescaleDB
```

---

## 5. State Machine

```
                    boot_complete()
  ┌──────┐  ─────────────────────────→  ┌──────────┐
  │ BOOT │                               │  ACTIVE  │
  └──────┘                               └────┬─────┘
                                               │
                              2 consecutive     │    any win or
                              losses            │    no bet
                                               │
                                    ┌──────────▼──────────┐
                                    │   COOLING_DOWN      │
                                    │   30 min timer       │
                                    └──────────┬──────────┘
                                               │
                              elapsed ≥ 1800s  │
                                               │
                                    ┌──────────▼──────────┐
                                    │     ACTIVE           │
                                    └─────────────────────┘
```

**During COOLING_DOWN**:
- ✅ Data feeds continue
- ✅ Features computed and persisted
- ✅ Model inference runs (logged only)
- ❌ Bet execution blocked
- ❌ Balance changes blocked

---

## 6. Compounding & Bet Sizing

### Bracket Table
| Index | Floor | Ceiling | Normal Bet | High-Conf Bet |
|-------|-------|---------|------------|---------------|
| 0 | $5.00 | $14.99 | $1.00 | $2.00 |
| 1 | $15.00 | $24.99 | $1.50 | $3.00 |
| 2 | $25.00 | $39.99 | $2.50 | $5.00 |
| 3 | $40.00 | $64.99 | $4.00 | $8.00 |
| 4 | $65.00 | $99.99 | $6.50 | $13.00 |
| 5 | $100.00 | ∞ | Kelly | Kelly |

### Upgrade Rules
- balance ≥ ceiling
- bets_in_bracket ≥ 10
- win_rate_in_bracket ≥ 50%
- If WR < 50% at ceiling → defer (don't upgrade)

### Downgrade Rules
- balance < floor → immediate downgrade
- Resets bracket stats

### Kelly Mode ($100+)
```
k   = (b × p − q) / b      (raw Kelly)
bet = min(k × 0.25 × balance, balance × 0.02)
```
- 25% fractional Kelly reduces variance
- 2% balance cap prevents overbet

### No Hard Stop
- Bot bets until balance < $1.00 (minimum bet)
- `resolve_bet_size()` returns 0.0 → execution skips

---

## 7. Feature Engine

**150+ features across 10 categories** computed in <100ms per 5-min candle.

All computations are **pure NumPy vectorized** — no Python for-loops on the hot path.
Numba JIT kernels used for: VWAP, OFI, Kyle's Lambda, Amihud Illiquidity.

**Critical constraint**: Feature computation reads from Ring Buffer only.
Never queries TimescaleDB during inference.

---

## 8. Signal Generation

### Thresholds (fixed, not tunable per session)
| Probability | Signal | Bet Size |
|-------------|--------|----------|
| P ≥ 0.82 | BET_UP_HIGH_CONF | High-conf bracket amount |
| P ≥ 0.72 | BET_UP_NORMAL | Normal bracket amount |
| 0.28 < P < 0.72 | NO_BET | — |
| P ≤ 0.28 | BET_DOWN_NORMAL | Normal bracket amount |
| P ≤ 0.18 | BET_DOWN_HIGH_CONF | High-conf bracket amount |

**No frequency cap** — consecutive signals all fire.
Cooling down is triggered by **loss outcomes**, not signal count.

---

## 9. Execution Engine

### 12-Step Cycle (every 5 minutes)
1. Wait for candle close
2. Compute features (always — state-agnostic)
3. Push features to DB queue
4. Check cooling down expiry
5. Gate: skip if not ACTIVE
6. Model inference → probability
7. Signal generation (with risk gates)
8. Market selection + odds EV check
9. Bet sizing
10. Paper or live execution
11. Wait for resolution
12. Update loss tracker + balance

### Paper vs Live
- Default: `paper_mode=True`
- Paper validation: ≥50 bets, >53% win rate required before live
- Live: needs `POLYMARKET_API_KEY` environment variable

---

## 10. Dashboard

**FastAPI backend** (port 8000) + **React 18 frontend** (port 3000)

### 11 Panels
1. Bot Status & Uptime (header strip)
2. Balance Overview (5 cards)
3. Total PnL (gross/fees/net)
4. Period Gain/Loss (daily/weekly/monthly)
5. Equity Curve (Recharts AreaChart)
6. Trade Statistics & Streaks
7. Performance by Side (BUY/SELL table)
8. Win Rate by Side (progress bars)
9. Performance by Hour UTC (bar chart)
10. Performance by Day of Week (bar chart)
11. Performance Calendar (monthly grid)

### Real-time Updates
- WebSocket `/ws/live` pushes every 5 seconds
- REST endpoints auto-refresh via React Query every 30 seconds

---

## 11. Monitoring & Alerting

### 10 Alert Rules
| Alert | Condition | Channel |
|-------|-----------|---------|
| Feed Down | No candle >2 min | Telegram + Email |
| Cooling Down Start | 2 consecutive losses | Telegram |
| Cooling Down End | Timer expired | Telegram |
| Hard Stop | Balance < $5.00 | Telegram + Email |
| Bracket Upgrade | Crossed ceiling | Telegram |
| Bracket Downgrade | Below floor | Telegram |
| Model Drift | PSI > 0.25 | Email |
| Win Rate Drop | 7-day WR < 50% | Email |
| API Error Rate | >5% in 1h | Telegram |
| Feature Latency | P95 > 100ms | Telegram |

### Feature Drift
- PSI (Population Stability Index) computed per feature
- PSI < 0.10 → OK, 0.10–0.25 → Warning, >0.25 → Retrain trigger

### Weekly Retrain
- Every Sunday 00:00 UTC
- Fetches 90 days of data
- Walk-forward validation
- Deploys only if accuracy improves by ≥0.5%

---

## 12. Testing Strategy

### Test Structure
```
tests/
├── unit/                    # 418+ unit tests
│   ├── test_ring_buffer.py
│   ├── test_features.py
│   ├── test_signal.py
│   ├── test_sizing.py
│   ├── test_cooling_down.py
│   ├── test_boot.py
│   ├── test_backtest.py
│   ├── test_execution.py
│   ├── test_dashboard.py
│   ├── test_monitoring.py
│   └── test_production_debug.py  ← Production hardening
├── integration/
│   └── test_boot_to_signal.py
└── perf/
    ├── test_feature_latency.py
    └── test_ingestion_performance.py
```

### Running Tests
```bash
# All logic tests (fast)
python -m pytest tests/ --ignore=tests/perf -q

# Performance tests (may fail under load)
python -m pytest tests/perf/ -v -s

# Production debugging only
python -m pytest tests/unit/test_production_debug.py -v
```

---

## 13. Production Debugging Notes

### Verified Safe ✅

**Race Conditions**: RingBuffer uses RLock for all writes. Concurrent push + read
tested under threading — no data corruption. StateManager's `check_cooling_down_expiry()`
is idempotent (calling it multiple times from different threads is harmless; only one
thread observes the True transition).

**Numerical Instability**: Kelly sizer clips probability to [0.001, 0.999] and validates
odds > 1.0. Returns 0.0 for negative-EV scenarios. Balance clamped to ≥0.0. Normalizer
uses epsilon=1e-10 for division safety. PSI uses epsilon=1e-8 for histogram bins.

**Cooling Down Behavior**: LossTracker resets counter to 0 upon triggering cooling down.
`_trigger_cooling_down()` guards with `is_active` check — losses during COOLING_DOWN
do NOT re-trigger. Counter accumulates fresh after expiry.

**Ring Buffer Edge Cases**: Empty buffer returns empty arrays. Wrap-around uses
np.concatenate of two slices. Unknown fields in push() silently ignored. Requesting
N > size returns only available bars. `update_latest` on empty buffer is a safe no-op.

**Model Degradation**: Signal thresholds use `>=` / `<=` comparisons (IEEE 754 compliant).
HIGH_CONF checked before NORMAL (hierarchy order). No floating-point boundary issues
observed at exact threshold values.

---

## 14. Configuration & Environment

### Required Environment Variables
```bash
# Binance (public data — no key needed for klines)
BINANCE_API_KEY=           # optional, for higher rate limits

# Polymarket (live mode only)
POLYMARKET_API_KEY=

# Telegram alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Email alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
ALERT_EMAIL_TO=

# Database
TIMESCALE_DB=btc_prediction
TIMESCALE_USER=btcbot
TIMESCALE_PASSWORD=btcbot_secret
TIMESCALE_PORT=5432
REDIS_PORT=6379
```

### Key Constants (in code)
| Constant | Value | Location |
|----------|-------|----------|
| CANDLE_INTERVAL_SEC | 300 | execution_engine.py |
| COOLING_DOWN_DURATION_SEC | 1800 | state_manager.py |
| CONSECUTIVE_LOSS_TRIGGER | 2 | loss_tracker.py |
| RING_BUFFER_CAPACITY | 600 | ring_buffer.py |
| KELLY_FRACTION | 0.25 | compounding_engine.py |
| KELLY_MAX_PCT | 0.02 | compounding_engine.py |
| MINIMUM_BET_FLOOR | 1.00 | compounding_engine.py |
| CIRCUIT_BREAKER_THRESHOLD | 5 | polymarket_client.py |
| PSI_WARNING_THRESHOLD | 0.10 | drift_detector.py |
| PSI_RETRAIN_THRESHOLD | 0.25 | drift_detector.py |
| RATE_LIMIT_SECONDS | 300 | alerting.py |

---

## 15. Deployment

### Docker Compose
```bash
docker-compose up -d         # TimescaleDB + Redis
```

### Running the Bot
```bash
# Paper mode (default)
python -m src.main

# Dashboard
uvicorn src.dashboard.api:app --port 8000 --reload
cd frontend && npm install && npm run dev
```

### Production Checklist
- [ ] All 469 tests pass
- [ ] Paper trading: ≥50 bets, >53% win rate
- [ ] Set POLYMARKET_API_KEY
- [ ] Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
- [ ] Set paper_mode=False in ExecutionEngine
- [ ] Monitor drift detector weekly
- [ ] Check dashboard daily
