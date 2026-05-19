# BTC 5-Minute Prediction Bot for Polymarket

AI-Powered Binary Options Prediction System targeting Polymarket's BTC 5-minute binary prediction markets.

**Test Status**: 469 tests passing | Phases 1–11 complete

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Node.js 18+ (for dashboard frontend)
- Binance API key (optional for public data)

### 1. Environment Setup
```bash
cd opus-1.0

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Infrastructure
```bash
# Start TimescaleDB, Redis, API, Dashboard
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Run Tests
```bash
# All logic tests (fast, ~18s)
python -m pytest tests/ --ignore=tests/perf -q

# Production debugging suite only
python -m pytest tests/unit/test_production_debug.py -v

# Performance tests (may vary by machine)
python -m pytest tests/perf/ -v -s
```

### 4. Run the Bot (Paper Mode)
```bash
python -m src.main
```

### 5. Run Dashboard
```bash
# Backend API
uvicorn src.dashboard.api:app --port 8000 --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

---

## Architecture

The system follows a strict **I/O / CPU separation** architecture:

```
I/O Layer (asyncio)  →  Ring Buffer (in-memory)  →  CPU Layer (thread pool)
       ↓                                                    ↓
  Persistence Queue                                  Feature Vectors
       ↓
  TimescaleDB (async, non-blocking)
```

### Key Design Principles
1. **Ring Buffer** — Pre-allocated numpy arrays (600 bars, ~0.14 MB) for zero-copy data retention
2. **I/O Separation** — WebSocket handlers never run numpy; feature engine never touches network
3. **Async Persistence** — DB writes are fire-and-forget from the hot path
4. **Always-On Features** — Feature engine runs during cooling down periods (no data gaps)
5. **No Hard Stop** — Bot bets until balance < $1.00 minimum bet

## Project Structure

```
src/
├── memory/          # Ring Buffer (pre-allocated numpy circular buffer)
├── ingestion/       # WebSocket, REST, derivatives clients
├── features/        # Feature engine (150+ features, <100ms)
├── models/          # LightGBM + LSTM + XGBoost ensemble
├── signal/          # Signal generation & risk gates
├── sizing/          # Compounding bracket system + Kelly sizer
├── execution/       # Polymarket CLOB integration + paper trader
├── backtest/        # Walk-forward backtesting engine
├── dashboard/       # FastAPI + React web dashboard
├── training/        # Model training pipeline + weekly retrain
├── monitoring/      # Alerting, drift detection, performance tracking
└── db/              # TimescaleDB schema & async client
frontend/
├── src/
│   ├── App.jsx          # Main dashboard layout (11 panels)
│   ├── App.css          # Premium dark theme
│   ├── components/      # Panel1–Panel11 components
│   └── hooks/           # useWebSocket auto-reconnect hook
└── package.json
tests/
├── unit/            # 469 unit tests
├── integration/     # Boot-to-signal integration tests
└── perf/            # Latency & throughput benchmarks
```

## Development Phases

| Phase | Description | Tests | Status |
|-------|-------------|-------|--------|
| 1 | Data Infrastructure | 52 | ✅ Complete |
| 2 | Feature Engineering | 67 | ✅ Complete |
| 3 | Model Development | 42 | ✅ Complete |
| 4 | Signal Generation | 38 | ✅ Complete |
| 5 | Compounding & Sizing | 36 | ✅ Complete |
| 6 | Cooling Down System | 22 | ✅ Complete |
| 7 | Fast Warm-Up & Boot | 18 | ✅ Complete |
| 8 | Backtesting | 37 | ✅ Complete |
| 9 | Live Trading | 34 | ✅ Complete |
| 10 | Web Dashboard | 31 | ✅ Complete |
| 11 | Monitoring & Alerting | 51 | ✅ Complete |
| — | Production Debugging | 51 | ✅ Complete |
| | **Total** | **469** | ✅ |

## Key Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)** — Full architecture, module reference, data flow, and production debugging notes
- **[PRD_BTC_Prediction_Bot.md](PRD_BTC_Prediction_Bot.md)** — Original Product Requirements Document

## Environment Variables

```bash
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

# Database (defaults work with docker-compose)
TIMESCALE_DB=btc_prediction
TIMESCALE_USER=btcbot
TIMESCALE_PASSWORD=btcbot_secret
```

## Production Checklist

- [ ] All 469 tests pass
- [ ] Paper trading validation: ≥50 bets, >53% win rate
- [ ] Set `POLYMARKET_API_KEY`
- [ ] Set `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
- [ ] Set `paper_mode=False` in ExecutionEngine
- [ ] Monitor drift detector weekly

## License

Private — All rights reserved.
