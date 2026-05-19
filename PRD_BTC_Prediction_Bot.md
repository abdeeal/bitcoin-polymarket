# PRD: BTC 5-Minute Prediction Bot for Polymarket
**Product Requirements Document — AI-Powered Binary Options Prediction System**

> **Version:** 3.0.0
> **Status:** Ready for Development
> **Last Updated:** 2026-05-17
> **Target Platform:** Polymarket (CLOB API)
> **Asset:** Bitcoin (BTC/USD)
> **Prediction Type:** Binary Up/Down — 5-Minute Horizon
> **Target Entry Rate:** 6–9% of candles (selective, high-confidence only)
> **Starting Capital:** $10 USDC
> **Bet Sizing:** Fixed bracket compounding ($1 normal / $2 high-confidence)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals & Success Metrics](#2-goals--success-metrics)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Phase 1 — Data Infrastructure](#4-phase-1--data-infrastructure)
5. [Phase 2 — Feature Engineering (High-Performance)](#5-phase-2--feature-engineering-high-performance)
6. [Phase 3 — Model Development](#6-phase-3--model-development)
7. [Phase 4 — Signal Generation & Filtering](#7-phase-4--signal-generation--filtering)
8. [Phase 5 — Compounding & Position Sizing](#8-phase-5--compounding--position-sizing)
9. [Phase 6 — Cooling Down System](#9-phase-6--cooling-down-system)
10. [Phase 7 — Fast Warm-Up & Boot Sequence](#10-phase-7--fast-warm-up--boot-sequence)
11. [Phase 8 — Backtesting & Validation](#11-phase-8--backtesting--validation)
12. [Phase 9 — Live Trading Integration](#12-phase-9--live-trading-integration)
13. [Phase 10 — Web Dashboard](#13-phase-10--web-dashboard)
14. [Phase 11 — Monitoring & Continuous Improvement](#14-phase-11--monitoring--continuous-improvement)
15. [Data Schema Reference](#15-data-schema-reference)
16. [Feature Catalog (150+ Features)](#16-feature-catalog-150-features)
17. [Tech Stack & Dependencies](#17-tech-stack--dependencies)
18. [Non-Functional Requirements](#18-non-functional-requirements)
19. [Risk & Constraints](#19-risk--constraints)
20. [Glossary](#20-glossary)

---

## 1. Executive Summary

### 1.1 Problem Statement

Polymarket offers binary prediction markets on BTC price direction within a 5-minute window. The market resolves as UP or DOWN relative to a target price. Manual trading is inefficient and emotionally biased. The opportunity lies in building an AI-driven system that:

- Ingests multi-source real-time market data with I/O and CPU load fully separated
- Engineers 150+ predictive features in **<100ms** using full vectorization, JIT compilation, and in-memory data retention
- Trains an ensemble ML model hot-loaded on boot using historical data — no waiting period
- Places bets **only** when model confidence is statistically significant (6–9% entry rate)
- Applies a compounding bracket system starting from $10 USDC
- Protects capital via a 30-minute cooling down pause triggered by 2 consecutive losses
- Continuously collects feature data even during cooling down to prevent cold-start on resume
- Provides a comprehensive web dashboard with 11 analytics panels for full observability

### 1.2 Core Hypothesis

Short-term BTC price movements, while largely random, contain **exploitable microstructure signals** — particularly in order book imbalance, trade flow, volume delta, and cross-asset divergence. By combining 150+ features computed with maximum computational efficiency, a high-confidence threshold filter, and disciplined compounding, the system can achieve a win rate above the break-even threshold consistently enough to generate positive EV from a $10 starting capital.

### 1.3 Scope

| In Scope | Out of Scope |
|---|---|
| BTC/USD 5-minute binary prediction | Other assets (ETH, SOL standalone) |
| Polymarket CLOB API integration | Other prediction platforms |
| Automated bet execution | Manual UI trading interface |
| Fixed bracket compounding from $10 | Portfolio-level multi-asset management |
| Cooling down system (2 consecutive losses) | Emotion-based manual overrides |
| Fast warm-up boot sequence | Real-time news NLP parsing |
| Full vectorization + JIT feature engine | On-chain blockchain data analysis |
| Separated I/O and CPU pipelines | |
| In-memory data retention | |
| Web dashboard (11 analytics panels) | |

---

## 2. Goals & Success Metrics

### 2.1 Primary Goals

- **G1:** Achieve >55% win rate on live predictions (break-even ~52% after fees)
- **G2:** Entry rate between 6–9% of total 5-minute candles
- **G3:** Positive expected value (EV) per bet over rolling 30-day window
- **G4:** Full automation: zero manual intervention for bet execution
- **G5:** System uptime >99% during active trading hours
- **G6:** Bot ready to place first bet within one 5-minute cycle of startup
- **G7:** Zero feature data gaps during cooling down periods
- **G8:** Feature computation latency <100ms using vectorization + JIT

### 2.2 Key Performance Indicators (KPIs)

| KPI | Target | Measurement |
|---|---|---|
| Win Rate | >55% | (Correct predictions / Total bets) × 100 |
| Entry Rate | 6–9% | (Bets placed / Total candles) × 100 |
| Sharpe Ratio | >1.5 | Rolling 30-day risk-adjusted return |
| Max Drawdown | <50% of bracket floor | Peak-to-trough within a bracket |
| Model Accuracy (backtest) | >57% | Walk-forward OOS accuracy |
| Feature Importance Stability | <10% rank drift | Top-20 feature rank change month-over-month |
| **Feature computation latency** | **<100ms** | Measured per 5m bar end-to-end |
| Total signal-to-bet latency | <500ms | Candle close to bet placed |
| Boot-to-ready (cold) | <3 minutes | System start to first eligible bet |
| Boot-to-ready (warm) | <45 seconds | |
| Feature gap during cooling down | 0% | Candles missed during cooling down |

### 2.3 Break-Even Analysis

```
Polymarket typical payout : ~1.9x on winning bet
Break-even win rate       : 1 / 1.9 = 52.6%
Target win rate           : 55–58%
EV per $1 bet at 56% WR  : 0.56 × $0.90 - 0.44 × $1.00 = +$0.064 (+6.4%)
EV per $2 bet at 56% WR  : 0.56 × $1.80 - 0.44 × $2.00 = +$0.128 (+6.4%)
Estimated bets/day        : ~20 (7% of 288 candles)
Combined daily EV         : ~$1.53/day theoretical from $10 capital
```

### 2.4 Compounding Projection (Theoretical, 56% WR)

```
Start   : $10.00
Week 1  : $10 → $12–$15
Month 1 : $10 → $25–$45
Month 3 : $45 → $100–$200
Month 6+: Switch to Kelly-based sizing at $100+
```

---

## 3. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   Binance    │  │  Order Book  │  │ Derivatives  │  │Polymarket │  │
│  │  WebSocket   │  │   L2 Feed    │  │  (Deribit)   │  │ CLOB API  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼─────────────────┼─────────────────┼────────────────┼────────┘
          │                 │                 │                │
┌─────────▼─────────────────▼─────────────────▼────────────────▼────────┐
│                     I/O LAYER (async, non-blocking)                    │
│   asyncio event loop | aiohttp | WebSocket handlers | REST pollers     │
│   Raw data → Ring Buffer (in-memory, no DB queries during inference)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │             BOOT SEQUENCE CHECK                  │
          │  Cold start? → REST backfill → populate buffers  │
          │  Model load from disk → warm in <3 min           │
          └────────────────────────┬────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│               CPU LAYER — FEATURE ENGINE (separate thread pool)        │
│   Full NumPy vectorization | Numba JIT-compiled kernels                │
│   In-memory circular buffers (no DB reads during compute)             │
│   Target: <100ms for all 150 features                                  │
│   ⚡ ALWAYS RUNNING — including during cooling down                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       ML INFERENCE ENGINE                               │
│        LightGBM + LSTM + Ensemble | P(price_up_5m)                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                   SIGNAL FILTER & RISK GATE                            │
│     Confidence Threshold | 5 Market Gates | Bot State Gate             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                    COOLING DOWN MANAGER                                 │
│   2 consecutive losses → 30-min execution pause                        │
│   Feature collection & DB writes CONTINUE during pause                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                     COMPOUNDING ENGINE                                  │
│   Fixed bracket sizing | $1 normal / $2 high-conf | Kelly at $100+     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                      EXECUTION ENGINE                                   │
│          Polymarket CLOB API | Bet Placement | Position Tracking        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┴──────────────────────┐
         │                                                 │
┌────────▼──────────────┐                    ┌────────────▼──────────────┐
│    PERSISTENCE LAYER  │                    │      WEB DASHBOARD        │
│  TimescaleDB (async)  │                    │  FastAPI + React/Svelte   │
│  Writes async, non-   │                    │  11 analytics panels      │
│  blocking from I/O    │                    │  WebSocket real-time feed │
└───────────────────────┘                    └───────────────────────────┘
```

---

## 4. Phase 1 — Data Infrastructure

### 4.1 Objectives

- Establish reliable, low-latency, multi-source data ingestion
- **Strict separation of I/O and CPU workloads** — WebSocket handlers never block compute; feature engine never touches network
- Raw data landed in **in-memory circular buffers** — feature engine reads from memory, never re-queries DB
- DB writes are **asynchronous and non-blocking** — fire-and-forget from the hot path
- Support REST backfill on cold boot for fast warm-up

### 4.2 I/O vs CPU Separation Architecture

```
┌─────────────────────────────────────────────────────────┐
│  I/O THREAD (asyncio event loop)                        │
│  - WebSocket recv handlers                              │
│  - REST API polling                                     │
│  - Writes raw ticks to Ring Buffer                      │
│  - Schedules async DB write (fire-and-forget)           │
│  NEVER: runs numpy, pandas, or ML code                  │
└────────────────────────┬────────────────────────────────┘
                         │  shared memory (lock-free ring buffer)
┌────────────────────────▼────────────────────────────────┐
│  CPU THREAD POOL (concurrent.futures or ProcessPool)    │
│  - Reads from Ring Buffer                               │
│  - Runs vectorized NumPy feature computation            │
│  - Runs Numba JIT kernels for hot paths                 │
│  - Feeds model inference                                │
│  NEVER: touches network, opens sockets, queries DB      │
└────────────────────────┬────────────────────────────────┘
                         │  async queue
┌────────────────────────▼────────────────────────────────┐
│  PERSISTENCE WORKER (background asyncio task)           │
│  - Drains feature vectors from async queue              │
│  - Bulk-inserts to TimescaleDB (batched, async)         │
│  - Non-blocking: does not affect I/O or CPU paths       │
└─────────────────────────────────────────────────────────┘
```

### 4.3 In-Memory Ring Buffer Design

```python
# File: src/memory/ring_buffer.py
# Fixed-size circular numpy arrays for zero-copy data retention
# All feature engine reads come from these buffers — no DB queries

import numpy as np
from threading import RLock

class RingBuffer:
    """
    Lock-free-read circular buffer backed by pre-allocated numpy arrays.
    Holds last N bars of OHLCV + order book + trade flow data.
    Size: 600 bars (50 hours) — covers all rolling window requirements.
    """
    CAPACITY = 600

    def __init__(self):
        # Pre-allocate all arrays at init — no runtime allocation
        self.open      = np.zeros(self.CAPACITY, dtype=np.float64)
        self.high      = np.zeros(self.CAPACITY, dtype=np.float64)
        self.low       = np.zeros(self.CAPACITY, dtype=np.float64)
        self.close     = np.zeros(self.CAPACITY, dtype=np.float64)
        self.volume    = np.zeros(self.CAPACITY, dtype=np.float64)
        self.taker_buy = np.zeros(self.CAPACITY, dtype=np.float64)
        self.timestamp = np.zeros(self.CAPACITY, dtype=np.int64)

        # Order book snapshots
        self.bid_depth = np.zeros(self.CAPACITY, dtype=np.float64)
        self.ask_depth = np.zeros(self.CAPACITY, dtype=np.float64)
        self.spread    = np.zeros(self.CAPACITY, dtype=np.float64)
        self.ofi       = np.zeros(self.CAPACITY, dtype=np.float64)

        # Derivatives
        self.funding   = np.zeros(self.CAPACITY, dtype=np.float64)
        self.oi        = np.zeros(self.CAPACITY, dtype=np.float64)
        self.liq_long  = np.zeros(self.CAPACITY, dtype=np.float64)
        self.liq_short = np.zeros(self.CAPACITY, dtype=np.float64)

        # Cross-asset
        self.eth_close = np.zeros(self.CAPACITY, dtype=np.float64)
        self.sol_close = np.zeros(self.CAPACITY, dtype=np.float64)

        self._head = 0     # points to oldest entry
        self._size = 0     # how many valid entries
        self._lock = RLock()

    def push(self, bar: dict) -> None:
        with self._lock:
            idx = self._head % self.CAPACITY
            # Write all fields in-place — no allocation
            self.close[idx]     = bar['close']
            self.open[idx]      = bar['open']
            self.high[idx]      = bar['high']
            self.low[idx]       = bar['low']
            self.volume[idx]    = bar['volume']
            self.taker_buy[idx] = bar['taker_buy']
            self.timestamp[idx] = bar['timestamp']
            # ... other fields
            self._head = (self._head + 1) % self.CAPACITY
            self._size = min(self._size + 1, self.CAPACITY)

    def get_window(self, n: int) -> dict[str, np.ndarray]:
        """Return last n bars as contiguous numpy arrays — zero copy."""
        with self._lock:
            n = min(n, self._size)
            end = self._head
            start = (end - n) % self.CAPACITY
            if start < end:
                return {
                    'close':  self.close[start:end],
                    'open':   self.open[start:end],
                    'high':   self.high[start:end],
                    'low':    self.low[start:end],
                    'volume': self.volume[start:end],
                    # ... etc
                }
            else:
                # Wrap-around: concatenate two slices
                return {
                    'close':  np.concatenate([self.close[start:], self.close[:end]]),
                    # ... etc
                }
```

### 4.4 Data Sources

#### 4.4.1 Primary: Binance WebSocket Streams

| Stream | Data | Frequency |
|---|---|---|
| `btcusdt@kline_1m` | OHLCV 1-minute candles | Per candle close |
| `btcusdt@depth20@100ms` | Order book L2 (20 levels) | 100ms |
| `btcusdt@aggTrade` | Aggregated trades | Per trade |
| `btcusdt@bookTicker` | Best bid/ask | Real-time |

#### 4.4.2 Historical Backfill: Binance REST API

| Endpoint | Data | Used For |
|---|---|---|
| `GET /api/v3/klines` | Historical OHLCV | Cold boot warm-up |
| `GET /fapi/v1/fundingRate` | Funding rate history | Feature backfill |
| `GET /fapi/v1/openInterest` | OI history | Feature backfill |

> **Boot Backfill Policy:** On cold start, fetch last **600 candles** (~10 hours) via REST before switching to WebSocket stream. Populate the Ring Buffer directly. Max rolling window is 200 bars — 600 bars ensures full coverage with headroom.

#### 4.4.3 Secondary: Derivatives Data

| Source | Data | Frequency |
|---|---|---|
| Binance Futures | Funding rate, OI | Every 5 minutes |
| Deribit REST API | IV, put/call ratio | Every 1 minute |
| Coinglass API | Liquidation data | Every 1 minute |

#### 4.4.4 Cross-Asset

| Source | Data | Frequency |
|---|---|---|
| Binance `ethusdt@kline_1m` | ETH OHLCV | Per candle |
| Binance `solusdt@kline_1m` | SOL OHLCV | Per candle |

#### 4.4.5 Polymarket

| Source | Data | Use |
|---|---|---|
| CLOB API `/markets` | Active BTC markets | Market discovery |
| CLOB API `/orderbook` | Current odds | Pre-bet EV check |
| CLOB API `/trade` | Bet placement | Execution |

### 4.5 Storage Architecture

```
In-Memory (always live — primary data source for feature engine)
├── RingBuffer (600 bars BTC OHLCV + OB + microstructure + derivatives)
├── RingBuffer (600 bars ETH/SOL cross-asset)
└── Redis (bot state, bracket, consecutive loss count, latest feature vector)

TimescaleDB (async persistence — never read during inference)
├── raw_candles_1m
├── raw_orderbook_snapshots
├── raw_trades
├── raw_funding_oi
├── raw_liquidations
├── features_5m              # Written async — bot_state column included
├── model_predictions
├── bet_log                  # Full bet + feature snapshot + outcome
└── cooling_down_log
```

### 4.6 Async Persistence Worker

```python
# File: src/db/persistence_worker.py
# Runs as background asyncio task
# Consumes from an asyncio.Queue — never called directly from hot path

class PersistenceWorker:
    def __init__(self, db_pool, queue: asyncio.Queue):
        self.db    = db_pool
        self.queue = queue

    async def run(self) -> None:
        batch = []
        while True:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(item)
                # Drain queue up to 50 items for bulk insert
                while not self.queue.empty() and len(batch) < 50:
                    batch.append(self.queue.get_nowait())
                await self.db.bulk_insert(batch)
                batch.clear()
            except asyncio.TimeoutError:
                if batch:
                    await self.db.bulk_insert(batch)
                    batch.clear()
```

### 4.7 Deliverables — Phase 1

- [ ] `src/memory/ring_buffer.py` — pre-allocated numpy circular buffer
- [ ] `src/ingestion/binance_ws.py` — WebSocket client, writes to Ring Buffer only
- [ ] `src/ingestion/binance_rest.py` — REST client for historical backfill
- [ ] `src/ingestion/derivatives_client.py` — REST polling, writes to Ring Buffer
- [ ] `src/ingestion/data_manager.py` — orchestrates all ingestion, separates I/O from CPU
- [ ] `src/db/persistence_worker.py` — async background writer
- [ ] `src/db/schema.sql` — TimescaleDB schema + hypertables + `cooling_down_log`
- [ ] `docker-compose.yml` — TimescaleDB + Redis containers
- [ ] Performance test: ingestion pipeline adds <5ms latency to Ring Buffer writes
- [ ] Integration test: 600 candle backfill completes in <90 seconds

---

## 5. Phase 2 — Feature Engineering (High-Performance)

### 5.1 Objectives

- Compute all 150+ features in **<100ms** per 5-minute bar
- **Full NumPy vectorization** — zero Python for-loops in any hot-path computation
- **Numba JIT compilation** for custom kernels (rolling stats, order flow imbalance, Kyle's lambda)
- **In-memory data only** — all reads from Ring Buffer, never from TimescaleDB during inference
- **CPU thread pool** — feature computation runs on separate thread from I/O event loop
- Feature engine runs continuously regardless of bot state (ACTIVE, COOLING_DOWN, BOOT)

### 5.2 Performance Budget per Feature Category

| Category | Features | Target Latency | Implementation |
|---|---|---|---|
| Price Action | 20 | <5ms | Pure NumPy vectorized |
| Volume | 15 | <5ms | Pure NumPy vectorized |
| Technical Indicators | 35 | <20ms | TA-Lib (C bindings) |
| Order Book | 20 | <15ms | NumPy + Numba JIT |
| Trade Flow / Microstructure | 15 | <20ms | Numba JIT |
| Cross-Asset & Regime | 12 | <10ms | Pure NumPy vectorized |
| Sentiment & Derivatives | 10 | <10ms | Pure NumPy vectorized |
| Time / Calendar | 8 | <1ms | Pure Python (trivial) |
| Lagged & Interaction | 15 | <10ms | Pure NumPy vectorized |
| Normalization (all features) | — | <5ms | NumPy vectorized |
| **TOTAL** | **~150** | **<100ms** | |

### 5.3 Vectorization Rules (Mandatory)

```python
# File: src/features/vectorization_standards.py
# ALL feature computation MUST follow these rules:

# ✅ CORRECT — fully vectorized
def compute_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(close) / close[:-1]          # single vectorized op

def compute_body_ratio(o, h, l, c):
    hl = h - l
    hl = np.where(hl == 0, 1e-10, hl)          # avoid div-zero, vectorized
    return (c - o) / hl

def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    # Use stride tricks for O(n) rolling — no Python loop
    shape  = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return windows.std(axis=-1)

# ❌ FORBIDDEN — Python for-loop on hot path
def bad_returns(close):
    ret = []
    for i in range(1, len(close)):   # NEVER do this
        ret.append((close[i] - close[i-1]) / close[i-1])
    return ret
```

### 5.4 Numba JIT Kernels (Custom Hot Paths)

```python
# File: src/features/jit_kernels.py
# Numba @njit for custom computations not covered by NumPy/TA-Lib

from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def compute_ofi(bid_sizes: np.ndarray, ask_sizes: np.ndarray,
                bid_prices: np.ndarray, ask_prices: np.ndarray) -> float:
    """Order Flow Imbalance — Numba JIT for speed."""
    ofi = 0.0
    for i in range(1, len(bid_sizes)):
        delta_bid = bid_sizes[i] - bid_sizes[i-1] if bid_prices[i] == bid_prices[i-1] else bid_sizes[i]
        delta_ask = ask_sizes[i] - ask_sizes[i-1] if ask_prices[i] == ask_prices[i-1] else -ask_sizes[i]
        ofi += delta_bid - delta_ask
    return ofi

@njit(cache=True, fastmath=True)
def compute_kyle_lambda(returns: np.ndarray, volumes: np.ndarray,
                        window: int) -> np.ndarray:
    """Kyle's Lambda (market impact) — rolling, JIT compiled."""
    n      = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan
    for i in range(window, n):
        ret_w = returns[i-window:i]
        vol_w = volumes[i-window:i]
        sqrt_vol = np.sqrt(vol_w)
        sqrt_vol = np.where(sqrt_vol == 0, 1e-10, sqrt_vol)
        result[i] = np.mean(np.abs(ret_w) / sqrt_vol)
    return result

@njit(cache=True, fastmath=True)
def compute_amihud(returns: np.ndarray, volumes: np.ndarray,
                   window: int) -> np.ndarray:
    """Amihud illiquidity ratio — rolling, JIT compiled."""
    n      = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan
    for i in range(window, n):
        vol_w = volumes[i-window:i]
        vol_w = np.where(vol_w == 0, 1e-10, vol_w)
        result[i] = np.mean(np.abs(returns[i-window:i]) / vol_w)
    return result

@njit(cache=True, fastmath=True)
def compute_vwap_rolling(close: np.ndarray, volume: np.ndarray,
                          window: int) -> np.ndarray:
    """Rolling VWAP — JIT compiled."""
    n      = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    for i in range(window - 1, n):
        pv  = close[i-window+1:i+1] * volume[i-window+1:i+1]
        vol = volume[i-window+1:i+1]
        result[i] = np.sum(pv) / (np.sum(vol) + 1e-10)
    return result

@njit(cache=True, fastmath=True)
def percentile_rank_rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank — JIT compiled."""
    n      = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan
    for i in range(window, n):
        w    = arr[i-window:i]
        rank = np.sum(w <= arr[i]) / window
        result[i] = rank
    return result
```

> **JIT Pre-compilation:** All `@njit` kernels must be pre-compiled at boot via a dummy call with a small array. This eliminates first-call JIT overhead (typically 200ms–2s) before the live trading loop begins. Boot sequence handles this automatically.

### 5.5 Feature Engine Architecture

```python
# File: src/features/feature_engine.py
# Reads from Ring Buffer (in-memory) — NEVER from DB
# Runs in CPU thread pool, separate from I/O event loop

import concurrent.futures
import numpy as np

class FeatureEngine:
    def __init__(self, ring_buffer: RingBuffer):
        self.buf       = ring_buffer
        self.executor  = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._precompile_jit_kernels()

    def _precompile_jit_kernels(self) -> None:
        """Warm up all Numba JIT kernels at init. Called once at boot."""
        dummy = np.random.rand(50).astype(np.float64)
        compute_ofi(dummy, dummy, dummy, dummy)
        compute_kyle_lambda(dummy, dummy, 10)
        compute_amihud(dummy, dummy, 10)
        compute_vwap_rolling(dummy, dummy, 10)
        percentile_rank_rolling(dummy, 10)

    async def compute_all(self, bot_state: str) -> dict:
        """
        Compute all 150 features from in-memory Ring Buffer.
        Runs in thread pool to avoid blocking asyncio event loop.
        Always executes — regardless of bot_state.
        """
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            self.executor, self._compute_sync, bot_state
        )
        return features

    def _compute_sync(self, bot_state: str) -> dict:
        """Synchronous feature computation — runs in thread pool."""
        # Single read from Ring Buffer — no further I/O
        w = self.buf.get_window(200)   # last 200 bars, all fields

        features = {}
        features.update(self._price_action(w))
        features.update(self._volume(w))
        features.update(self._technical(w))
        features.update(self._order_book(w))
        features.update(self._microstructure(w))
        features.update(self._cross_asset(w))
        features.update(self._sentiment(w))
        features.update(self._calendar())
        features.update(self._lagged_interaction(w, features))
        features['bot_state'] = bot_state
        return features
```

### 5.6 Technical Indicators — TA-Lib Integration

```python
# File: src/features/technical.py
# TA-Lib uses C extensions — fast, but API takes arrays, not scalars
# Must pass full window arrays and take last value for current bar

import talib

def compute_technical(w: dict) -> dict:
    c = w['close'].astype(np.float64)  # ensure contiguous float64
    h = w['high'].astype(np.float64)
    l = w['low'].astype(np.float64)

    return {
        # RSI — vectorized over full window, take last value
        'rsi_7':  talib.RSI(c, timeperiod=7)[-1],
        'rsi_14': talib.RSI(c, timeperiod=14)[-1],
        'rsi_21': talib.RSI(c, timeperiod=21)[-1],

        # MACD
        'macd_line':      talib.MACD(c, 12, 26, 9)[0][-1],
        'macd_signal':    talib.MACD(c, 12, 26, 9)[1][-1],
        'macd_histogram': talib.MACD(c, 12, 26, 9)[2][-1],

        # Bollinger Bands
        'bb_upper': talib.BBANDS(c, 20, 2)[0][-1],
        'bb_lower': talib.BBANDS(c, 20, 2)[2][-1],

        # ATR
        'atr_7':  talib.ATR(h, l, c, timeperiod=7)[-1],
        'atr_14': talib.ATR(h, l, c, timeperiod=14)[-1],

        # EMA
        'ema_5':  talib.EMA(c, 5)[-1],
        'ema_10': talib.EMA(c, 10)[-1],
        'ema_20': talib.EMA(c, 20)[-1],
        'ema_50': talib.EMA(c, 50)[-1],

        # Stochastic
        'stoch_k_14': talib.STOCH(h, l, c, 14, 3, 0, 3, 0)[0][-1],
        'stoch_d_14': talib.STOCH(h, l, c, 14, 3, 0, 3, 0)[1][-1],

        # CCI + Williams
        'cci_14':       talib.CCI(h, l, c, 14)[-1],
        'williams_r_14': talib.WILLR(h, l, c, 14)[-1],
    }
```

### 5.7 Normalization — Vectorized Rolling Z-Score

```python
# File: src/features/normalizer.py
# All normalization done via vectorized ops on in-memory arrays
# Rolling stats maintained in Ring Buffer — never recomputed from scratch

def zscore_normalize(value: float, arr: np.ndarray, window: int) -> float:
    """Single-bar z-score using pre-computed rolling window — no loop."""
    w    = arr[-window:]
    mean = w.mean()
    std  = w.std()
    return (value - mean) / (std + 1e-10)

def minmax_normalize(value: float, arr: np.ndarray, window: int) -> float:
    w    = arr[-window:]
    mn   = w.min()
    mx   = w.max()
    rng  = mx - mn
    return (value - mn) / (rng + 1e-10)
```

### 5.8 Performance Benchmarking Requirements

```python
# File: tests/perf/test_feature_latency.py
# Must pass before deployment

def test_feature_engine_latency():
    engine = FeatureEngine(mock_ring_buffer(600))
    times  = []
    for _ in range(100):     # 100 iterations to get stable estimate
        t0 = time.perf_counter()
        asyncio.run(engine.compute_all('ACTIVE'))
        times.append(time.perf_counter() - t0)
    p95 = np.percentile(times, 95) * 1000   # convert to ms
    assert p95 < 100, f"P95 feature latency {p95:.1f}ms exceeds 100ms budget"
```

### 5.9 Deliverables — Phase 2

- [ ] `src/memory/ring_buffer.py` — pre-allocated numpy circular buffer
- [ ] `src/features/jit_kernels.py` — all Numba @njit kernels with `cache=True`
- [ ] `src/features/price_action.py` — fully vectorized
- [ ] `src/features/volume.py` — fully vectorized
- [ ] `src/features/technical.py` — TA-Lib C bindings
- [ ] `src/features/order_book.py` — NumPy + Numba JIT
- [ ] `src/features/microstructure.py` — Numba JIT for OFI, Kyle, Amihud
- [ ] `src/features/cross_asset.py` — fully vectorized
- [ ] `src/features/sentiment.py` — fully vectorized
- [ ] `src/features/calendar.py` — trivial
- [ ] `src/features/lagged_interaction.py` — fully vectorized
- [ ] `src/features/normalizer.py` — rolling z-score + minmax, vectorized
- [ ] `src/features/feature_engine.py` — async orchestrator, thread pool, state-agnostic
- [ ] `tests/perf/test_feature_latency.py` — P95 <100ms test
- [ ] No Python for-loops in any hot-path file (enforced via `flake8` custom rule)

---

## 6. Phase 3 — Model Development

### 6.1 Objectives

- Train and validate ensemble model achieving >57% accuracy on walk-forward OOS
- Implement proper time-series cross-validation (no data leakage)
- Tune model to produce well-calibrated probability outputs
- Model serialized for fast load on boot (<30 seconds)

### 6.2 Target Variable Definition

```python
def compute_label(df: pd.DataFrame) -> pd.Series:
    # 1 if close price 5 minutes later > current close, else 0
    return (df['close'].shift(-5) > df['close']).astype(int)
    # Rows where shift(-5) is NaN excluded from training
```

### 6.3 Model Architecture

#### 6.3.1 LightGBM (Baseline — weight 0.50)

```python
# File: src/models/lgbm_model.py
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 63,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}
```

#### 6.3.2 LSTM (Temporal — weight 0.35)

```python
# File: src/models/lstm_model.py
# Input  : sequence of last 30 5-minute bars × 150 features
# Layer 1: LSTM(128) + Dropout(0.3)
# Layer 2: LSTM(64)  + Dropout(0.3)
# Layer 3: Dense(32) + ReLU
# Output : Dense(1) + Sigmoid
# Optimizer: Adam (lr=0.001)
# Epochs: 100 with early stopping (patience=10)
```

#### 6.3.3 XGBoost (Diversity — weight 0.15)

```python
# File: src/models/xgb_model.py
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 800,
    'learning_rate': 0.025,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'tree_method': 'hist',   # fast histogram method
    'random_state': 42,
    'n_jobs': -1
}
```

#### 6.3.4 Ensemble

```python
# File: src/models/ensemble.py
class EnsembleModel:
    def predict_proba(self, X) -> float:
        p_lgb  = self.lgbm.predict_proba(X)[:, 1]
        p_lstm = self.lstm.predict(X_seq)
        p_xgb  = self.xgb.predict_proba(X)[:, 1]
        return 0.50 * p_lgb + 0.35 * p_lstm + 0.15 * p_xgb

    def save(self, path: str): ...   # <10s save, <30s load
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel': ...
```

### 6.4 Walk-Forward Validation Protocol

```
Training window : Rolling 90-day window
Validation      : Next 10 days (OOS, strict chronological)
Step            : Retrain every 10 days
Minimum bars    : 2,592 (90 days × 288 bars/day)
```

### 6.5 Hyperparameter Tuning

```python
# File: src/training/hyperopt_search.py
# Optuna, 200 trials, TimeSeriesSplit(n_splits=5)
# Objective: maximize OOS log-loss
```

### 6.6 Model Calibration

```python
# File: src/models/calibration.py
# Platt scaling (CalibratedClassifierCV, method='sigmoid', cv=5)
# Target ECE < 0.03
```

### 6.7 Deliverables — Phase 3

- [ ] `src/models/lgbm_model.py`, `lstm_model.py`, `xgb_model.py`, `ensemble.py`, `calibration.py`
- [ ] `src/training/train_pipeline.py`, `walk_forward.py`, `hyperopt_search.py`
- [ ] `notebooks/01_eda.ipynb`, `02_feature_importance.ipynb`, `03_walk_forward_results.ipynb`
- [ ] Model load time test: <30 seconds

---

## 7. Phase 4 — Signal Generation & Filtering

### 7.1 Confidence Thresholding

```python
# File: src/signal/signal_generator.py

class SignalGenerator:
    THRESHOLD_UP_NORMAL   = 0.72   # $1 bet — ~6-7% of candles
    THRESHOLD_DOWN_NORMAL = 0.28
    THRESHOLD_UP_HIGH     = 0.82   # $2 bet — ~1-2% of candles
    THRESHOLD_DOWN_HIGH   = 0.18

    # HIGH-CONF: no frequency cap — consecutive signals all execute
    # Cooling down triggered by OUTCOMES, not signal frequency

    def generate_signal(self, prob: float, filters_ok: bool) -> Signal:
        if not filters_ok:
            return Signal.NO_BET
        if prob >= self.THRESHOLD_UP_HIGH:
            return Signal.BET_UP_HIGH_CONF
        elif prob <= self.THRESHOLD_DOWN_HIGH:
            return Signal.BET_DOWN_HIGH_CONF
        elif prob >= self.THRESHOLD_UP_NORMAL:
            return Signal.BET_UP_NORMAL
        elif prob <= self.THRESHOLD_DOWN_NORMAL:
            return Signal.BET_DOWN_NORMAL
        return Signal.NO_BET
```

### 7.2 Risk Gate Filters

```python
# File: src/signal/risk_gates.py

# Gate 1 — Volatility Regime
def volatility_gate(f) -> bool:
    return f['vol_regime'] < 2.0 and f['atr_percentile'] < 90

# Gate 2 — Spread
def spread_gate(f) -> bool:
    return f['spread_bps'] < f['spread_bps_20m_avg'] * 2.5

# Gate 3 — BB Squeeze
def squeeze_gate(f) -> bool:
    return not f['bb_squeeze']

# Gate 4 — Liquidation Cascade
def liquidation_gate(f) -> bool:
    return (f['liq_long_1m'] + f['liq_short_1m']) < f['liq_5m_avg'] * 5.0

# Gate 5 — Polymarket Odds EV
def odds_gate(signal, odds, model_prob) -> bool:
    return (model_prob - 1 / odds) > 0.05

# Gate 6 — Bot State (blocks execution during cooling down)
def bot_state_gate(state: BotState) -> bool:
    return state == BotState.ACTIVE
```

### 7.3 Deliverables — Phase 4

- [ ] `src/signal/signal_generator.py` — NORMAL and HIGH_CONF tiers, no freq cap on high-conf
- [ ] `src/signal/risk_gates.py` — all 6 gates
- [ ] `src/signal/odds_checker.py`
- [ ] Unit tests: consecutive high-conf signals all fire, gate logic correct

---

## 8. Phase 5 — Compounding & Position Sizing

### 8.1 Fixed Bracket System

```python
# File: src/sizing/compounding_engine.py

BRACKETS = [
    # index, floor,   ceil,   normal, high_conf
    (0,    5.00,   14.99,  1.00,  2.00),   # START
    (1,   15.00,   24.99,  1.50,  3.00),
    (2,   25.00,   39.99,  2.50,  5.00),
    (3,   40.00,   64.99,  4.00,  8.00),
    (4,   65.00,   99.99,  6.50, 13.00),
    (5,  100.00,   np.inf, None,  None),    # Kelly mode
]

HARD_STOP_BALANCE = 5.00
```

### 8.2 Bracket Upgrade Rules

```python
def can_upgrade(self) -> bool:
    return (
        self.balance >= self.current_bracket.ceil
        and self.bets_in_bracket >= 10          # minimum sample
        and self.win_rate_in_bracket >= 0.50    # positive record required
    )
# If win rate < 50% at ceiling → defer upgrade, log warning
```

### 8.3 Bracket Downgrade Rules

```python
def check_downgrade(self) -> None:
    if self.balance < self.current_bracket.floor:
        self.current_bracket = self.find_bracket(self.balance)
        self.bets_in_bracket = 0
        self.win_rate_in_bracket = 0.0

def check_hard_stop(self) -> bool:
    if self.balance < HARD_STOP_BALANCE:
        self.set_state(BotState.PAUSED)
        alert.send("HARD STOP: balance < $5.00")
        return True
    return False
```

### 8.4 Bet Size Resolution

```python
def resolve_bet_size(self, signal: Signal) -> float:
    if self.balance >= 100.00:
        return self.kelly_size(signal)
    bracket = self.current_bracket
    if signal in [Signal.BET_UP_HIGH_CONF, Signal.BET_DOWN_HIGH_CONF]:
        return bracket.high_conf_bet
    return bracket.normal_bet

def kelly_size(self, signal: Signal) -> float:
    b  = self.last_odds - 1
    q  = 1 - self.last_prob
    k  = (b * self.last_prob - q) / b
    return round(min(k * 0.25 * self.balance, self.balance * 0.02), 2)
```

### 8.5 Deliverables — Phase 5

- [ ] `src/sizing/compounding_engine.py`
- [ ] `src/sizing/portfolio_tracker.py`
- [ ] `src/sizing/kelly_sizer.py`
- [ ] Unit + simulation tests

---

## 9. Phase 6 — Cooling Down System

### 9.1 Rationale

Two consecutive losses indicate the model may be misreading the current regime — attempting a reversal that isn't happening, or chasing a continuation in a shifted market. A 30-minute pause allows the regime to stabilize. Since feature collection never stops, the model resumes with 6 fresh, uninterrupted candles of context.

### 9.2 State Machine

```
ACTIVE → COOLING_DOWN (trigger: consecutive_loss_count == 2)
  → reset counter to 0
  → record cooling_down_started_at
  → block bet execution via bot_state_gate
  → continue feature collection and DB writes
  → send Telegram: "🔴 COOLING DOWN — 2 consecutive losses. 30 min pause."

COOLING_DOWN → ACTIVE (trigger: elapsed >= 1800 seconds)
  → set bot_state = ACTIVE
  → log ended_at to cooling_down_log
  → send Telegram: "🟢 Cooling down ended — bot resumed."
  → consecutive_loss_count stays 0 (fresh start)
```

### 9.3 Consecutive Loss Counter

```python
# File: src/execution/loss_tracker.py
class LossTracker:
    def on_bet_resolved(self, won: bool) -> None:
        if won:
            self.consecutive_losses = 0     # any win fully resets counter
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 2:
                self._trigger_cooling_down()

    def _trigger_cooling_down(self) -> None:
        state_manager.set_state(BotState.COOLING_DOWN)
        redis.set('cooling_down_started_at', datetime.utcnow().isoformat())
        redis.set('consecutive_loss_count', 0)
        db.insert_cooling_down_log(...)
        alert.telegram("🔴 COOLING DOWN: 2 consecutive losses...")

    def check_expiry(self) -> None:
        if state_manager.state != BotState.COOLING_DOWN:
            return
        elapsed = (datetime.utcnow() - cooling_start).total_seconds()
        if elapsed >= 1800:
            state_manager.set_state(BotState.ACTIVE)
            db.update_cooling_down_log(ended_at=datetime.utcnow())
            alert.telegram("🟢 Cooling down ended — bot resumed.")
```

### 9.4 Behavior Matrix During Cooling Down

| Component | During Cooling Down |
|---|---|
| WebSocket feeds | ✅ Fully active |
| Ring Buffer writes | ✅ Continuous |
| Feature engine | ✅ Computes & persists every 5m |
| Model inference | ✅ Runs (logged only, not executed) |
| Signal generator | ✅ Generates (logged only) |
| Bet execution | ❌ Blocked by bot_state_gate |
| Compounding engine | ❌ No balance changes |
| Cooling down timer | ✅ Counting down |

### 9.5 Deliverables — Phase 6

- [ ] `src/execution/loss_tracker.py`
- [ ] `src/execution/state_manager.py`
- [ ] `cooling_down_log` table in schema
- [ ] Integration test: feature rows written during full simulated cooling down period
- [ ] Alert test: Telegram messages on start and end

---

## 10. Phase 7 — Fast Warm-Up & Boot Sequence

### 10.1 Boot Flow

```
System start
│
▼ Step 1 — Load model from disk (models/latest/)
  Target: <30s | Fallback: trigger training if no model found
│
▼ Step 2 — Pre-compile all Numba JIT kernels (dummy call)
  Target: <10s | Prevents first-call latency during live trading
│
▼ Step 3 — Check Ring Buffer / DB for existing data
  If last candle < 1 hour ago → WARM START (skip backfill)
  Else → COLD START → Step 4
│
▼ Step 4 (cold only) — REST backfill: fetch 600 candles + ETH/SOL + funding + OI
  Write directly to Ring Buffer AND async DB
  Target: <90s
│
▼ Step 5 — Run Feature Engine on Ring Buffer history
  Compute features for last 200 bars (batch mode)
  Populate Redis latest_features
  Target: <60s
│
▼ Step 6 — Connect all WebSocket streams (live)
  Target: <5s
│
▼ Step 7 — Set bot_state = ACTIVE
  Load balance + bracket from DB/Redis
  Reset consecutive_loss_count = 0
  Log boot_completed_at
│
▼ READY — wait for next 5-minute candle close (<5 min)
  Bot will place first bet if signal fires on next close
```

### 10.2 Boot Timing Targets

| Step | Cold Start | Warm Start |
|---|---|---|
| Model load | <30s | <30s |
| JIT pre-compile | <10s | <10s |
| REST backfill | <90s | — |
| Feature computation | <60s | <20s |
| WebSocket connect | <5s | <5s |
| **Total** | **<3 min** | **<45s** |

### 10.3 Boot Sequence Spec

```python
# File: src/boot/boot_sequence.py
class BootSequence:
    async def run(self) -> None:
        self.model      = EnsembleModel.load('models/latest/')
        self.feature_engine._precompile_jit_kernels()  # JIT warmup
        needs_backfill  = await self._check_data_freshness()
        if needs_backfill:
            await self.data_manager.backfill_historical(n_candles=600)
        await self._batch_compute_features(n_bars=200)
        await self.data_manager.connect_all_streams()
        redis.set('bot_state', BotState.ACTIVE)
        redis.set('consecutive_loss_count', 0)
        balance = await self.portfolio_tracker.load_balance()
        bracket = self.compounding_engine.find_bracket(balance)
        redis.set('current_bracket', bracket.index)
```

### 10.4 Deliverables — Phase 7

- [ ] `src/boot/boot_sequence.py`
- [ ] `src/boot/warmup_validator.py`
- [ ] Integration test: cold start places first bet within 5-minute window
- [ ] Integration test: JIT kernels pre-compiled before first live bar

---

## 11. Phase 8 — Backtesting & Validation

### 11.1 Backtest Engine with Cooling Down Simulation

```python
# File: src/backtest/backtester.py
class Backtester:
    def __init__(self, start_date, end_date, initial_capital=10.0):
        self.capital            = initial_capital
        self.compounding        = CompoundingEngine(initial_capital)
        self.consecutive_losses = 0
        self.cooling_down_until = None
        self.bet_log            = []

    def run(self, feature_df: pd.DataFrame, model: EnsembleModel):
        for i, bar in feature_df.iterrows():
            # Features always computed (mirrors production)
            features = bar[FEATURE_COLS]

            # Cooling down check — skip execution only, not data
            in_cd = (self.cooling_down_until is not None
                     and bar['timestamp'] < self.cooling_down_until)
            if in_cd:
                continue

            prob   = model.predict_proba(features)
            signal = signal_gen.generate_signal(prob, filters_ok=True)
            if signal == Signal.NO_BET:
                continue

            bet_size = self.compounding.resolve_bet_size(signal)
            won      = self._simulate_outcome(bar, signal)
            pnl      = net_payout(bet_size, bar['odds'], won)
            self.capital += pnl
            self.compounding.update_balance(self.capital)

            if not won:
                self.consecutive_losses += 1
                if self.consecutive_losses >= 2:
                    self.cooling_down_until = bar['timestamp'] + timedelta(minutes=30)
                    self.consecutive_losses = 0
            else:
                self.consecutive_losses = 0

            self.bet_log.append({
                'timestamp': bar['timestamp'],
                'signal': signal.name,
                'prob': prob,
                'bet_size': bet_size,
                'won': won,
                'pnl': pnl,
                'balance': self.capital,
            })
```

### 11.2 Cost Simulation

```python
POLYMARKET_FEE = 0.02
SLIPPAGE_BPS   = 5

def net_payout(bet_size, odds, won):
    if won:
        gross = bet_size * (odds - 1)
        return gross - gross * POLYMARKET_FEE
    return -bet_size
```

### 11.3 Anti-Leakage Checklist

- [ ] Labels use `shift(-5)` — future data never visible to features
- [ ] Normalization uses only data up to `t`
- [ ] Walk-forward splits strictly chronological
- [ ] Cooling down skips execution, not data collection

### 11.4 Deliverables — Phase 8

- [ ] `src/backtest/backtester.py` — full simulation with cooling down
- [ ] `src/backtest/cost_model.py`, `report_generator.py`, `leakage_detector.py`
- [ ] `notebooks/04_backtest_results.ipynb`
- [ ] Minimum 6-month OOS backtest passing before live

---

## 12. Phase 9 — Live Trading Integration

### 12.1 Polymarket CLOB Integration

```python
# File: src/execution/polymarket_client.py
class PolymarketClient:
    async def get_active_btc_markets(self) -> list[Market]
    async def get_market_odds(self, market_id: str) -> float
    async def place_bet(self, market_id, side, size, price) -> BetReceipt
    # Retry up to 3× with exponential backoff
    async def get_bet_status(self, order_id: str) -> BetStatus
```

### 12.2 Main Execution Loop

```python
# File: src/execution/execution_engine.py
async def execute_cycle():
    await wait_for_candle_close()

    # Always compute features — state-agnostic
    features = await feature_engine.compute_all(state_manager.state)

    # Push feature vector to async DB queue (non-blocking)
    persistence_queue.put_nowait(('features', features))

    # Check cooling down expiry
    loss_tracker.check_expiry()

    # Gate: only proceed if ACTIVE
    if state_manager.state != BotState.ACTIVE:
        return

    if compounding.check_hard_stop():
        return

    prob   = model.predict_proba(features)
    signal = signal_gen.generate_signal(prob, risk_gates.check_all(features))
    if signal == Signal.NO_BET:
        return

    market = await market_selector.find_best_market(signal)
    if not market or not odds_gate.check(signal, market.odds, prob):
        return

    bet_size = compounding.resolve_bet_size(signal)

    if PAPER_MODE:
        receipt = paper_trader.simulate(market, signal, bet_size)
    else:
        receipt = await polymarket_client.place_bet(
            market.id, signal.to_side(), bet_size, market.best_price
        )

    won = await wait_for_resolution(receipt)
    loss_tracker.on_bet_resolved(won)
    compounding.update_balance(portfolio.current_balance)
    persistence_queue.put_nowait(('bet', {**receipt, 'won': won, 'features': features}))
```

### 12.3 Paper Trading Validation Criteria

```
Minimum 2 weeks paper trading before going live.
Pass criteria:
  ✅ Win rate > 53% over 50+ paper bets
  ✅ Entry rate confirmed 6–9%
  ✅ ≥1 cooling down event triggered and recovered cleanly
  ✅ ≥1 bracket upgrade observed
  ✅ Latency < 500ms consistently
  ✅ Zero execution bugs in 2-week window
```

### 12.4 Deliverables — Phase 9

- [ ] `src/execution/polymarket_client.py`, `market_selector.py`, `paper_trader.py`
- [ ] `src/execution/execution_engine.py`, `loss_tracker.py`, `state_manager.py`, `bet_logger.py`
- [ ] Circuit breaker: pause on 5 consecutive API failures
- [ ] 2-week paper trading validation

---

## 13. Phase 10 — Web Dashboard

### 13.1 Overview

A full web dashboard providing real-time and historical visibility into every aspect of the bot. Built with FastAPI (backend WebSocket + REST) and a modern frontend (React or Svelte). Data served from TimescaleDB via async queries. Real-time updates pushed via WebSocket every 5 seconds.

### 13.2 Architecture

```
Frontend (React / Svelte)
  ├── WebSocket connection → real-time tick updates (every 5s)
  └── REST API calls → historical data on panel load

Backend (FastAPI)
  ├── GET /api/stats/summary         → panels 1–3
  ├── GET /api/stats/period          → panel 4
  ├── GET /api/stats/equity-curve    → panel 5
  ├── GET /api/stats/streaks         → panel 6
  ├── GET /api/stats/by-side         → panels 7–8
  ├── GET /api/stats/by-hour         → panel 9
  ├── GET /api/stats/by-day          → panel 10
  ├── GET /api/stats/by-calendar     → panel 11
  ├── GET /api/system/status         → bot state, cooling down
  └── WS  /ws/live                   → real-time updates stream

Database
  └── TimescaleDB — all queries async via asyncpg
```

### 13.3 Panel Specifications

---

#### Panel 1 — Bot Status & Uptime

**Location:** Top bar / header strip
**Data source:** Redis (real-time) + `bot_session_log`

```
┌──────────────────────────────────────────────────────────────┐
│  🟢 BOT ACTIVE   |   Total Uptime: 47d 12h 33m              │
│  Mode: LIVE      |   Current State: ACTIVE                   │
│  Last bet: 14:35 UTC (UP, won)                               │
│  Cooling down: —  (last: 3h ago, duration 30m)              │
└──────────────────────────────────────────────────────────────┘
```

**Fields:**
- `bot_state` — ACTIVE | COOLING_DOWN | PAUSED | BOOT
- `total_uptime_seconds` — sum of all active session durations
- `uptime_formatted` — "47d 12h 33m"
- `last_bet_timestamp`, `last_bet_direction`, `last_bet_outcome`
- `cooling_down_active` — bool
- `cooling_down_remaining_seconds` — if active
- `last_cooling_down_started_at`, `last_cooling_down_duration_sec`

**DB Query:**
```sql
SELECT
    SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, NOW()) - started_at))) AS total_uptime
FROM bot_session_log;
```

---

#### Panel 2 — Balance Overview

**Location:** Top row cards
**Data source:** `bet_log` + `portfolio_state`

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Starting Bal │ │ Current Bal  │ │ Total Growth │ │ Highest Bal  │ │ Lowest Bal   │
│   $10.00     │ │   $38.42     │ │  +$28.42     │ │   $41.87     │ │    $7.53     │
│              │ │              │ │  (+284.2%)   │ │  2026-04-12  │ │  2026-03-22  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

**Fields:**
- `starting_balance` — first recorded balance (from config or first bet_log row)
- `current_balance` — latest balance in portfolio_tracker
- `total_growth_usd` — current - starting
- `total_growth_pct` — (current - starting) / starting × 100
- `highest_balance` — max balance_after in bet_log
- `highest_balance_date`
- `lowest_balance` — min balance_after in bet_log
- `lowest_balance_date`

**DB Query:**
```sql
SELECT
    MIN(balance_before) FILTER (WHERE timestamp = (SELECT MIN(timestamp) FROM bet_log)) AS starting_balance,
    MAX(balance_after)  AS highest_balance,
    MIN(balance_after)  AS lowest_balance,
    (SELECT balance_after FROM bet_log ORDER BY timestamp DESC LIMIT 1) AS current_balance
FROM bet_log;
```

---

#### Panel 3 — Total PnL

**Location:** Top row card
**Data source:** `bet_log`

```
┌─────────────────────────────────┐
│  Total PnL                      │
│  +$28.42  (+284.2%)             │
│  Realized: $28.42               │
│  Fees paid: $2.14               │
│  Net: $26.28                    │
└─────────────────────────────────┘
```

**Fields:**
- `total_pnl_gross` — SUM(pnl) from bet_log
- `total_fees` — SUM of fees paid on winning bets
- `total_pnl_net` — gross - fees
- `total_pnl_pct` — relative to starting balance

---

#### Panel 4 — Period Gain/Loss

**Location:** Second row, wide card
**Data source:** `bet_log` aggregated by period

```
┌─────────────────────────────────────────────────────────────────┐
│  Period Performance                                             │
│  ┌──────────┬──────────┬───────────┐                           │
│  │ Daily    │ Weekly   │ Monthly   │                           │
│  │ +$1.24   │ +$7.83   │ +$18.42   │                           │
│  │ (+3.3%)  │ (+25.6%) │ (+92.1%)  │                           │
│  └──────────┴──────────┴───────────┘                           │
│  [Toggle: USD | %]   [Period selector: today / this week / MTD]│
└─────────────────────────────────────────────────────────────────┘
```

**Fields:**
- Daily: `pnl_today_usd`, `pnl_today_pct`, `pnl_yesterday_usd`
- Weekly: `pnl_this_week_usd`, `pnl_last_week_usd`
- Monthly: `pnl_this_month_usd`, `pnl_last_month_usd`

**DB Query:**
```sql
SELECT
    SUM(pnl) FILTER (WHERE timestamp >= CURRENT_DATE)                       AS pnl_today,
    SUM(pnl) FILTER (WHERE timestamp >= date_trunc('week', NOW()))          AS pnl_week,
    SUM(pnl) FILTER (WHERE timestamp >= date_trunc('month', NOW()))         AS pnl_month
FROM bet_log
WHERE outcome IS NOT NULL;
```

---

#### Panel 5 — Equity Curve

**Location:** Main chart area
**Data source:** `bet_log` ordered by timestamp

```
┌─────────────────────────────────────────────────────────────────────┐
│  Equity Curve                                              [7d|30d|All]
│                                                                     │
│  $42 ┤                                                    ╭──       │
│  $35 ┤                                        ╭──────────╯          │
│  $28 ┤                           ╭────────────╯                     │
│  $21 ┤              ╭────────────╯                                   │
│  $14 ┤  ╭───────────╯                                               │
│  $10 ┼──╯                                                           │
│      └──────────────────────────────────────────────────────────    │
│       Mar         Apr          May                                  │
│                                                                     │
│  [Bracket upgrade markers shown as vertical dotted lines]           │
│  [Cooling down periods shown as gray shaded regions]                │
└─────────────────────────────────────────────────────────────────────┘
```

**Fields:**
- Time series: `[{timestamp, balance_after}]` — one point per bet
- `bracket_change_timestamps` — vertical line markers
- `cooling_down_periods` — `[{start, end}]` for shaded regions

**DB Query:**
```sql
SELECT timestamp, balance_after
FROM bet_log
WHERE outcome IS NOT NULL
ORDER BY timestamp ASC;
```

---

#### Panel 6 — Trade Statistics & Streaks

**Location:** Stats row
**Data source:** `bet_log`

```
┌──────────────────────────────────────────────────────────────────────┐
│  Trade Statistics                                                    │
│                                                                      │
│  Total Entries : 312    Total Wins : 181    Total Losses : 131       │
│  Win Rate      : 58.0%                                               │
│                                                                      │
│  Max Win Streak  : 9   (Apr 14–16)    Max Lose Streak : 4           │
│  Curr Win Streak : 3                  Curr Lose Streak: 0           │
│                                                                      │
│  Avg Win  : +$0.78    Avg Loss : -$1.00    Profit Factor : 1.79     │
│  Largest Win : +$1.80  Largest Loss : -$2.00                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Fields:**
- `total_entries`, `total_wins`, `total_losses`, `win_rate`
- `max_win_streak`, `max_win_streak_start`, `max_win_streak_end`
- `max_lose_streak`, `max_lose_streak_start`
- `current_win_streak`, `current_lose_streak`
- `avg_win_usd`, `avg_loss_usd`, `profit_factor`
- `largest_win_usd`, `largest_loss_usd`

**Streak Computation (Python, not SQL):**
```python
def compute_streaks(outcomes: list[bool]) -> dict:
    """Vectorized streak computation using numpy."""
    arr     = np.array(outcomes, dtype=np.int8)   # 1=win, 0=loss
    changes = np.diff(arr, prepend=arr[0] + 1)    # detect changes
    groups  = np.split(arr, np.where(changes != 0)[0])
    streaks = [(bool(g[0]), len(g)) for g in groups if len(g) > 0]

    win_streaks  = [l for v, l in streaks if v]
    loss_streaks = [l for v, l in streaks if not v]

    current       = streaks[-1] if streaks else (True, 0)
    return {
        'max_win_streak':    max(win_streaks, default=0),
        'max_lose_streak':   max(loss_streaks, default=0),
        'current_win_streak':  current[1] if current[0] else 0,
        'current_lose_streak': current[1] if not current[0] else 0,
    }
```

---

#### Panel 7 — Performance by Side (Buy vs Sell)

**Location:** Stats row, card
**Data source:** `bet_log` grouped by signal direction

```
┌──────────────────────────────────────────────────┐
│  Performance by Side                             │
│                                                  │
│  Side    │ Entries │ PnL       │ Avg PnL         │
│  ─────────────────────────────────────────────── │
│  BUY ▲   │   167   │  +$16.42  │  +$0.098        │
│  SELL ▼  │   145   │  +$12.00  │  +$0.083        │
│  ─────────────────────────────────────────────── │
│  Total   │   312   │  +$28.42  │  +$0.091        │
└──────────────────────────────────────────────────┘
```

**DB Query:**
```sql
SELECT
    CASE WHEN signal LIKE 'BET_UP%' THEN 'BUY' ELSE 'SELL' END AS side,
    COUNT(*)             AS entries,
    SUM(pnl)             AS total_pnl,
    AVG(pnl)             AS avg_pnl
FROM bet_log
WHERE outcome IS NOT NULL
GROUP BY side;
```

---

#### Panel 8 — Win Rate by Side

**Location:** Stats row, card
**Data source:** `bet_log` grouped by signal direction

```
┌──────────────────────────────────────────────────┐
│  Win Rate by Side                                │
│                                                  │
│  BUY  ▲   58.1%  ████████████░░░░░░░░ (97/167)  │
│  SELL ▼   57.9%  ███████████░░░░░░░░░ (84/145)  │
│                                                  │
│  [Progress bar: win rate fill]                   │
└──────────────────────────────────────────────────┘
```

**DB Query:**
```sql
SELECT
    CASE WHEN signal LIKE 'BET_UP%' THEN 'BUY' ELSE 'SELL' END AS side,
    COUNT(*) FILTER (WHERE outcome = true)  AS wins,
    COUNT(*)                                AS total,
    ROUND(COUNT(*) FILTER (WHERE outcome = true)::numeric / COUNT(*) * 100, 1) AS win_rate_pct
FROM bet_log
WHERE outcome IS NOT NULL
GROUP BY side;
```

---

#### Panel 9 — Performance by Hour (1–24 UTC)

**Location:** Bar chart, full width
**Data source:** `bet_log` grouped by UTC hour

```
┌────────────────────────────────────────────────────────────────────────┐
│  Performance by Hour (UTC)                      [PnL $ | Win Rate %]  │
│                                                                        │
│  +$3 ┤         ██        ██                                           │
│  +$2 ┤  ██     ██  ██    ██  ██                                       │
│  +$1 ┤  ██  ██ ██  ██ ██ ██  ██  ██         ██                       │
│   $0 ┼──────────────────────────────────────────────────────────      │
│  -$1 ┤                              ░░  ░░                            │
│      └───────────────────────────────────────────────────────         │
│       1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
│                                                                        │
│  Hover: hour 14 UTC → 42 bets | PnL +$3.12 | Win rate 61.9%          │
└────────────────────────────────────────────────────────────────────────┘
```

**Fields per hour:** `hour`, `total_bets`, `total_pnl`, `win_rate`, `avg_pnl`

**DB Query:**
```sql
SELECT
    EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') + 1 AS hour,
    COUNT(*)                                AS total_bets,
    SUM(pnl)                                AS total_pnl,
    AVG(pnl)                                AS avg_pnl,
    ROUND(AVG(outcome::int) * 100, 1)       AS win_rate_pct
FROM bet_log
WHERE outcome IS NOT NULL
GROUP BY hour
ORDER BY hour;
```

---

#### Panel 10 — Performance by Day of Week

**Location:** Bar chart
**Data source:** `bet_log` grouped by weekday

```
┌────────────────────────────────────────────────────────┐
│  Performance by Day                   [PnL $ | Win %]  │
│                                                        │
│  +$8 ┤          ██                                    │
│  +$6 ┤  ██      ██  ██                                │
│  +$4 ┤  ██  ██  ██  ██      ██                        │
│  +$2 ┤  ██  ██  ██  ██  ██  ██  ██                    │
│   $0 ┼────────────────────────────────                │
│  -$2 ┤                          ░░                    │
│      └────────────────────────────────                │
│       Mon Tue Wed Thu Fri Sat Sun                      │
│                                                        │
│  Hover: Wednesday → 54 bets | PnL +$7.82 | WR 63.0%  │
└────────────────────────────────────────────────────────┘
```

**DB Query:**
```sql
SELECT
    TO_CHAR(timestamp AT TIME ZONE 'UTC', 'Dy') AS day_name,
    EXTRACT(DOW FROM timestamp)                  AS day_num,
    COUNT(*)                                     AS total_bets,
    SUM(pnl)                                     AS total_pnl,
    ROUND(AVG(outcome::int) * 100, 1)            AS win_rate_pct
FROM bet_log
WHERE outcome IS NOT NULL
GROUP BY day_name, day_num
ORDER BY day_num;
```

---

#### Panel 11 — Performance Calendar

**Location:** Full-width calendar view
**Data source:** `bet_log` grouped by calendar date

```
┌──────────────────────────────────────────────────────────────────────┐
│  Calendar Performance                    [May 2026]  ◀ ▶            │
│                                                                      │
│  Mon    Tue    Wed    Thu    Fri    Sat    Sun                        │
│  ────   ────   ────   ────   ────   ────   ────                      │
│                              1      2      3                         │
│                           +$1.24 -$0.50  +$2.10                     │
│   4      5      6      7      8      9     10                        │
│ +$0.80 +$1.50 +$2.30  +$0.10 -$1.20 +$0.90 +$1.70                  │
│  11     12     13     14     15     16     17                        │
│ +$0.60 +$3.10 +$1.20 +$0.90  TODAY  —      —                        │
│                                                                      │
│  Color: 🟢 green = profit day | 🔴 red = loss day | ⬜ = no bets     │
│  Hover on day: date, bets count, total PnL, win rate                 │
│  [Month navigator] [Year selector]                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Fields per day:** `date`, `total_bets`, `total_pnl`, `win_rate`, `is_profit_day`

**DB Query:**
```sql
SELECT
    DATE(timestamp AT TIME ZONE 'UTC')   AS trade_date,
    COUNT(*)                             AS total_bets,
    SUM(pnl)                             AS total_pnl,
    ROUND(AVG(outcome::int) * 100, 1)    AS win_rate_pct,
    SUM(pnl) > 0                         AS is_profit_day
FROM bet_log
WHERE outcome IS NOT NULL
  AND timestamp >= date_trunc('month', NOW())  -- filter by selected month
GROUP BY trade_date
ORDER BY trade_date;
```

---

### 13.4 Real-Time Update Spec

```python
# File: src/dashboard/ws_broadcaster.py
# FastAPI WebSocket endpoint pushes updates every 5 seconds

class WSBroadcaster:
    async def broadcast_loop(self):
        while True:
            payload = {
                'bot_state':           redis.get('bot_state'),
                'current_balance':     portfolio.current_balance,
                'consecutive_losses':  redis.get('consecutive_loss_count'),
                'cooling_down_active': state_manager.state == BotState.COOLING_DOWN,
                'cooling_down_remaining': self._remaining_cd_seconds(),
                'last_bet':            await db.get_last_bet(),
                'today_pnl':           await db.get_today_pnl(),
                'current_win_streak':  self.streaks.current_win,
                'current_lose_streak': self.streaks.current_lose,
            }
            await self.broadcast(json.dumps(payload))
            await asyncio.sleep(5)
```

### 13.5 Dashboard Tech Stack

| Component | Technology |
|---|---|
| Frontend | React 18 + Recharts (charts) + TailwindCSS |
| Backend API | FastAPI + asyncpg |
| Real-time | WebSocket (FastAPI native) |
| Charts | Recharts (equity curve, bar charts), custom calendar component |
| Data fetching | React Query (auto-refresh) |
| Hosting | Docker container, same compose stack |

### 13.6 Deliverables — Phase 10

- [ ] `src/dashboard/api.py` — FastAPI app with all 9 REST endpoints
- [ ] `src/dashboard/ws_broadcaster.py` — real-time WebSocket broadcaster
- [ ] `src/dashboard/queries.py` — all TimescaleDB queries (async)
- [ ] `frontend/src/App.jsx` — main layout with 11 panels
- [ ] `frontend/src/components/Panel1_Status.jsx`
- [ ] `frontend/src/components/Panel2_Balance.jsx`
- [ ] `frontend/src/components/Panel3_TotalPnl.jsx`
- [ ] `frontend/src/components/Panel4_PeriodGain.jsx`
- [ ] `frontend/src/components/Panel5_EquityCurve.jsx`
- [ ] `frontend/src/components/Panel6_Streaks.jsx`
- [ ] `frontend/src/components/Panel7_BySide.jsx`
- [ ] `frontend/src/components/Panel8_WinRateSide.jsx`
- [ ] `frontend/src/components/Panel9_ByHour.jsx`
- [ ] `frontend/src/components/Panel10_ByDay.jsx`
- [ ] `frontend/src/components/Panel11_Calendar.jsx`
- [ ] `frontend/src/hooks/useWebSocket.js` — real-time connection
- [ ] Docker: dashboard served on port 3000, API on port 8000
- [ ] All panels refresh via WebSocket — no manual page reload needed

---

## 14. Phase 11 — Monitoring & Continuous Improvement

### 14.1 Alerting Rules (Telegram + Email)

| Alert | Condition | Channel |
|---|---|---|
| Feed Down | No candle for >2 min | Telegram + Email |
| Cooling Down Start | 2 consecutive losses | Telegram |
| Cooling Down End | 30-min pause complete | Telegram |
| Hard Stop | Balance < $5.00 | Telegram + Email |
| Bracket Upgrade | Balance crossed ceiling | Telegram |
| Bracket Downgrade | Balance dropped below floor | Telegram |
| Model Drift | PSI > 0.25 on top 10 features | Email |
| Win Rate Drop | 7-day WR < 50% | Email |
| API Error Rate | >5% failed calls in 1h | Telegram |
| Feature Latency | P95 > 100ms | Telegram |

### 14.2 Weekly Retraining

```python
# File: src/training/retrain_scheduler.py
# Every Sunday 00:00 UTC
def weekly_retrain(self):
    df      = db.fetch_training_data(days=90)
    metrics = WalkForward(df).run(EnsembleModel)
    if metrics['accuracy'] > current_accuracy + 0.005:
        model = train_final_model(df)
        model.save('models/latest/')
        alert.send(f"New model deployed: acc={metrics['accuracy']:.4f}")
    normalizer.update_windows(df)
```

### 14.3 Feature Drift Detection

```python
# File: src/monitoring/drift_detector.py
# PSI per feature: >0.1 warning | >0.25 → retrain trigger
def compute_psi(reference, current, bins=10) -> float:
    ref = np.histogram(reference, bins=bins)[0] / len(reference) + 1e-8
    cur = np.histogram(current,   bins=bins)[0] / len(current)   + 1e-8
    return np.sum((cur - ref) * np.log(cur / ref))
```

### 14.4 Deliverables — Phase 11

- [ ] `src/monitoring/alerting.py`, `drift_detector.py`, `performance_tracker.py`
- [ ] `src/training/retrain_scheduler.py`
- [ ] MLflow experiment tracking integration

---

## 15. Data Schema Reference

### 15.1 `raw_candles_1m`

```sql
CREATE TABLE raw_candles_1m (
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
SELECT create_hypertable('raw_candles_1m', 'timestamp');
```

### 15.2 `features_5m`

```sql
CREATE TABLE features_5m (
    timestamp  TIMESTAMPTZ NOT NULL,
    label      SMALLINT,           -- 1=up, 0=down, NULL=pending
    bot_state  TEXT,               -- ACTIVE | COOLING_DOWN | BOOT
    features   JSONB,
    PRIMARY KEY (timestamp)
);
```

### 15.3 `bet_log`

```sql
CREATE TABLE bet_log (
    id               UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp        TIMESTAMPTZ NOT NULL,
    market_id        TEXT NOT NULL,
    signal           TEXT NOT NULL,       -- BET_UP_NORMAL | BET_UP_HIGH_CONF | etc.
    confidence_tier  TEXT NOT NULL,       -- NORMAL | HIGH_CONF
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
SELECT create_hypertable('bet_log', 'timestamp');
```

### 15.4 `cooling_down_log`

```sql
CREATE TABLE cooling_down_log (
    id               SERIAL PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ,
    trigger_reason   TEXT NOT NULL DEFAULT '2_consecutive_losses',
    duration_sec     INTEGER,
    balance_at_start NUMERIC(10,2),
    bracket_at_start INTEGER,
    candles_missed   INTEGER DEFAULT 0   -- should always be 0
);
```

### 15.5 `bot_session_log`

```sql
CREATE TABLE bot_session_log (
    id         SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at   TIMESTAMPTZ,
    reason     TEXT    -- 'normal_shutdown' | 'crash' | 'hard_stop'
);
```

---

## 16. Feature Catalog (150+ Features)

### 16.1 Price Action (20)

| Feature | Description | Type | Computation |
|---|---|---|---|
| `ret_1m` | 1-minute return | Float | `np.diff(close)[-1] / close[-2]` |
| `ret_2m` | 2-minute return | Float | Vectorized |
| `ret_3m` | 3-minute return | Float | Vectorized |
| `ret_5m` | 5-minute return | Float | Vectorized |
| `ret_10m` | 10-minute return | Float | Vectorized |
| `ret_15m` | 15-minute return | Float | Vectorized |
| `ret_30m` | 30-minute return | Float | Vectorized |
| `body_ratio` | (close-open)/(high-low) | Float | Vectorized |
| `upper_shadow` | Upper wick ratio | Float | Vectorized |
| `lower_shadow` | Lower wick ratio | Float | Vectorized |
| `candle_direction` | +1 green / -1 red | Int | `np.sign(close-open)` |
| `gap_open` | Open vs prev close | Float | Vectorized |
| `is_doji` | abs(body) < 0.1 | Binary | Vectorized |
| `consecutive_green` | Count last N green | Int | `np.cumsum` trick |
| `consecutive_red` | Count last N red | Int | `np.cumsum` trick |
| `close_vs_high` | (high-close)/(high-low) | Float | Vectorized |
| `close_vs_low` | (close-low)/(high-low) | Float | Vectorized |
| `hl_range_pct` | (high-low)/close | Float | Vectorized |
| `price_acceleration` | ret_1m - ret_2m | Float | Vectorized |
| `higher_high` | high > prev_high | Binary | Vectorized |

### 16.2 Volume (15)

| Feature | Description | Type |
|---|---|---|
| `volume_1m` | Raw 1m volume | Float |
| `volume_ratio_5m` | vol_1m / avg_vol_5m | Float |
| `volume_ratio_15m` | vol_1m / avg_vol_15m | Float |
| `volume_ratio_30m` | vol_1m / avg_vol_30m | Float |
| `volume_spike` | vol > 2× avg | Binary |
| `volume_trend` | Slope of volume last 5 bars | Float |
| `pv_corr_10m` | Price-volume corr 10 bars | Float |
| `vwap_5m` | VWAP 5m (Numba JIT) | Float |
| `vwap_15m` | VWAP 15m (Numba JIT) | Float |
| `close_vs_vwap5` | close - vwap_5m | Float |
| `close_vs_vwap15` | close - vwap_15m | Float |
| `vol_delta_approx` | Buy vol - sell vol estimate | Float |
| `cum_delta_5m` | Cumulative delta 5 bars | Float |
| `vol_imbalance` | (buy-sell)/total | Float |
| `up_vol_ratio` | Vol on green / total last 10 | Float |

### 16.3 Technical Indicators (35)

*(All via TA-Lib C bindings — full array input, take last value)*

RSI (7, 14, 21), RSI slope, RSI divergence, MACD line/signal/hist/slope/crossover, BB upper/lower/width/position/squeeze/breakout up/down, EMA (5, 10, 20, 50), close_vs_ema5, close_vs_ema20, ema5_vs_ema20, ma_alignment, ATR (7, 14), atr_ratio, normalized_range, atr_percentile, Stochastic K/D, stoch_position, CCI, Williams %R

### 16.4 Order Book (20)

*(NumPy + Numba JIT for hot paths)*

bid_ask_spread, spread_bps, bid_depth_1pct, ask_depth_1pct, depth_imbalance (0.1%, 0.5%, 1%, 2%), bid_ask_ratio (5 levels, 10 levels), large_bid_wall, large_ask_wall, book_imbalance_change, spread_change, mid_price_drift, quote_intensity, bid_depth_ratio_5m, ask_depth_ratio_5m, book_pressure, level2_absorption

### 16.5 Microstructure (15)

*(Numba JIT for OFI, Kyle's Lambda, Amihud)*

trade_count_1m, trade_count_ratio, large_trade_count, taker_buy_ratio_1m, taker_buy_ratio_5m, taker_buy_change, avg_trade_size_1m, avg_trade_size_ratio, whale_activity, kyle_lambda (JIT), amihud_illiquidity (JIT), ofi_1m (JIT), ofi_5m, trade_intensity_change, aggressive_buy_ratio

### 16.6 Cross-Asset & Regime (12)

eth_ret_5m, eth_btc_divergence, sol_ret_5m, crypto_market_ret, futures_basis, basis_change, funding_rate, funding_rate_8h_avg, funding_extreme, realized_vol_1h, realized_vol_24h, vol_regime

### 16.7 Sentiment & Derivatives (10)

liq_long_1m, liq_short_1m, liq_ratio, liq_spike, oi_change_5m, oi_change_15m, price_oi_diverge, put_call_ratio, iv_25delta, iv_change_1h

### 16.8 Time / Calendar (8)

hour_of_day, day_of_week, is_weekend, is_asian_session, is_london_session, is_ny_session, session_overlap, minutes_to_hour

### 16.9 Lagged & Interaction (15)

*(All NumPy vectorized)*

ret_lag1/2/3, direction_lag1/2, rolling_win_rate_20, rolling_ret_std_10, mean_reversion_score, rsi_x_volume, momentum_x_regime, bb_position_x_rsi, spread_x_volatility, depth_x_momentum, taker_x_regime, ofi_x_oi

---

## 17. Tech Stack & Dependencies

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Async framework | asyncio + aiohttp | Latest |
| JIT Compiler | Numba | 0.59+ |
| Array computation | NumPy | 1.26+ |
| Technical indicators | TA-Lib (C bindings) | 0.4.x |
| Database | TimescaleDB (PostgreSQL) | 2.x |
| Async DB driver | asyncpg | 0.29+ |
| Cache | Redis | 7.x |
| ML — Boosting | LightGBM | 4.x |
| ML — Boosting alt | XGBoost | 2.x |
| ML — Deep Learning | PyTorch | 2.x |
| Data processing | Pandas | 2.x |
| Hyperparameter tuning | Optuna | 3.x |
| Experiment tracking | MLflow | 2.x |
| Dashboard backend | FastAPI | 0.110+ |
| Dashboard frontend | React 18 + Recharts + Tailwind | Latest |
| Alerting | python-telegram-bot | Latest |
| Containerization | Docker + Docker Compose | Latest |
| Task scheduling | APScheduler | Latest |
| Code quality | flake8 + custom no-loop rule | Latest |

### 17.1 Directory Structure

```
btc-prediction-bot/
├── src/
│   ├── boot/
│   │   ├── boot_sequence.py
│   │   └── warmup_validator.py
│   ├── memory/
│   │   └── ring_buffer.py             # Pre-allocated numpy circular buffer
│   ├── ingestion/
│   │   ├── binance_ws.py              # I/O only — writes to RingBuffer
│   │   ├── binance_rest.py            # REST backfill
│   │   ├── derivatives_client.py
│   │   └── data_manager.py
│   ├── features/
│   │   ├── jit_kernels.py             # All @njit kernels (cache=True)
│   │   ├── feature_engine.py          # Async orchestrator, CPU thread pool
│   │   ├── price_action.py            # Pure NumPy
│   │   ├── volume.py                  # Pure NumPy
│   │   ├── technical.py               # TA-Lib
│   │   ├── order_book.py              # NumPy + Numba
│   │   ├── microstructure.py          # Numba JIT
│   │   ├── cross_asset.py             # Pure NumPy
│   │   ├── sentiment.py               # Pure NumPy
│   │   ├── calendar.py
│   │   ├── lagged_interaction.py      # Pure NumPy
│   │   └── normalizer.py              # Vectorized rolling stats
│   ├── models/
│   │   ├── lgbm_model.py
│   │   ├── lstm_model.py
│   │   ├── xgb_model.py
│   │   ├── ensemble.py
│   │   └── calibration.py
│   ├── signal/
│   │   ├── signal_generator.py
│   │   ├── risk_gates.py
│   │   └── odds_checker.py
│   ├── sizing/
│   │   ├── compounding_engine.py
│   │   ├── portfolio_tracker.py
│   │   └── kelly_sizer.py
│   ├── execution/
│   │   ├── polymarket_client.py
│   │   ├── market_selector.py
│   │   ├── paper_trader.py
│   │   ├── execution_engine.py
│   │   ├── loss_tracker.py
│   │   ├── state_manager.py
│   │   └── bet_logger.py
│   ├── backtest/
│   │   ├── backtester.py
│   │   ├── cost_model.py
│   │   ├── report_generator.py
│   │   └── leakage_detector.py
│   ├── dashboard/
│   │   ├── api.py                     # FastAPI app (11 endpoints)
│   │   ├── ws_broadcaster.py          # Real-time WebSocket
│   │   └── queries.py                 # All async TimescaleDB queries
│   ├── training/
│   │   ├── train_pipeline.py
│   │   ├── walk_forward.py
│   │   ├── hyperopt_search.py
│   │   └── retrain_scheduler.py
│   ├── monitoring/
│   │   ├── alerting.py
│   │   ├── drift_detector.py
│   │   └── performance_tracker.py
│   └── db/
│       ├── schema.sql
│       ├── client.py                  # asyncpg pool
│       ├── persistence_worker.py      # Async background writer
│       └── migrations/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Panel1_Status.jsx
│   │   │   ├── Panel2_Balance.jsx
│   │   │   ├── Panel3_TotalPnl.jsx
│   │   │   ├── Panel4_PeriodGain.jsx
│   │   │   ├── Panel5_EquityCurve.jsx
│   │   │   ├── Panel6_Streaks.jsx
│   │   │   ├── Panel7_BySide.jsx
│   │   │   ├── Panel8_WinRateSide.jsx
│   │   │   ├── Panel9_ByHour.jsx
│   │   │   ├── Panel10_ByDay.jsx
│   │   │   └── Panel11_Calendar.jsx
│   │   └── hooks/
│   │       └── useWebSocket.js
│   └── package.json
├── models/
│   └── latest/
├── notebooks/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── perf/
│       └── test_feature_latency.py    # P95 <100ms assertion
├── config/
│   ├── live.yaml
│   ├── paper.yaml
│   └── backtest.yaml
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.frontend
├── requirements.txt
├── .env.example
└── README.md
```

---

## 18. Non-Functional Requirements

### 18.1 Performance

| Requirement | Target | Enforcement |
|---|---|---|
| Feature computation P95 | <100ms | `test_feature_latency.py` |
| Feature computation P99 | <150ms | Monitored, alerted |
| Model inference | <100ms | Measured per cycle |
| Total signal-to-bet | <500ms | Measured per cycle |
| Data feed freshness | <50ms | WebSocket handler benchmark |
| No Python for-loops on hot path | Zero | `flake8` custom lint rule |
| Ring Buffer read | <1ms | Pre-allocated numpy |
| DB write blocking main loop | Never | Async queue worker |
| Cold boot to ACTIVE | <3 min | Boot integration test |
| Warm boot to ACTIVE | <45s | Boot integration test |

### 18.2 I/O and CPU Separation Enforcement

- I/O thread (asyncio loop): **only** handles WebSocket recv, REST calls, writing to Ring Buffer, scheduling async DB writes
- CPU thread pool: **only** reads from Ring Buffer, runs numpy/numba/talib, feeds model
- Persistence worker (async task): **only** drains write queue, bulk-inserts to DB
- **Cross-boundary rule:** no numpy/talib calls in I/O loop; no network calls in CPU thread pool — enforced via code review checklist

### 18.3 Reliability

- WebSocket auto-reconnect with exponential backoff
- Circuit breaker: auto-halt on 5 consecutive API failures
- Data gap detection: alert if >2 consecutive candles missing
- Feature engine continues on partial data (missing derivatives = use last known value, not null)
- Write-ahead logging on all bet operations (prevent duplicate bets)
- Ring Buffer never loses data on reconnect (WebSocket restores missed bars via REST catch-up)

### 18.4 Security

- API keys: environment variables only — never committed to git
- Polymarket private key: encrypted at rest
- All external calls: HTTPS/WSS
- Bet size hard cap: enforced at execution layer independently of sizing layer

---

## 19. Risk & Constraints

### 19.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Feature latency regression after code change | Medium | High | `test_feature_latency.py` in CI |
| Numba JIT compilation on first live call | Low | High | Pre-compile at boot (dummy call) |
| Ring Buffer overflow (>600 bars between flushes) | Very Low | Low | Buffer sized 10× max window |
| I/O/CPU boundary violation introduced in refactor | Medium | High | Code review checklist + lint |
| WebSocket drop during bet cycle | Medium | High | Auto-reconnect + state check |
| Model performance decay | High | High | Weekly retrain + drift detection |
| Data leakage in training | Medium | Critical | Automated leakage checker |
| Feature gap during cooling down | Very Low | High | Feature engine unconditionally runs |

### 19.2 Market Risks

| Risk | Mitigation |
|---|---|
| Flash crash / black swan | Volatility gate + liquidation gate |
| Sustained losing streak | Cooling down + bracket downgrade + hard stop |
| Regime change | vol_regime feature + weekly retrain |
| Edge decay over time | Drift detection + retrain pipeline |

### 19.3 Constraints

- Polymarket minimum bet: $1 USDC — bracket floor is $1
- Starting capital: $10 USDC (Bracket 0)
- Hard stop floor: $5 USDC
- Binance WebSocket limit: 5 connections per IP — use combined stream endpoint
- Deribit API rate limit: 10 req/s

---

## 20. Glossary

| Term | Definition |
|---|---|
| Ring Buffer | Pre-allocated fixed-size numpy array used as circular buffer — primary in-memory data store |
| JIT | Just-In-Time compilation via Numba `@njit` — compiles Python to native machine code |
| Vectorization | Computing on entire arrays with NumPy ops instead of Python for-loops |
| I/O Layer | asyncio event loop handling all network operations — no CPU-heavy work |
| CPU Layer | Thread pool running feature computation — no network access |
| Persistence Worker | Background async task draining write queue to TimescaleDB — non-blocking |
| CLOB | Central Limit Order Book — Polymarket's order matching system |
| OFI | Order Flow Imbalance — computed via Numba JIT kernel |
| PSI | Population Stability Index — feature drift metric |
| Walk-Forward | Strict chronological validation — no data leakage |
| Kelly Criterion | Optimal bet sizing formula — used at $100+ balance |
| Cooling Down | 30-minute execution pause triggered by 2 consecutive losses |
| Cold Start | Boot with no existing DB data — requires REST backfill |
| Warm Start | Boot with recent DB data — skips backfill |
| Bracket | Balance range with fixed normal ($1) and high-conf ($2) bet sizes |
| Hard Stop | Balance < $5.00 → all betting paused until manual review |
| High-Conf Signal | P > 0.82 or P < 0.18 — $2 bet, no frequency cap |
| Normal Signal | P > 0.72 or P < 0.28 — $1 bet |
| Equity Curve | Time series of balance_after — displayed in Panel 5 |
| Profit Factor | Total gross profit / total gross loss |

---

## Appendix A — Development Timeline

| Phase | Description | Duration | Key Constraint |
|---|---|---|---|
| 1 | Data Infrastructure + Ring Buffer + I/O separation | 2 weeks | Foundation for all phases |
| 2 | Feature Engineering (vectorized + JIT) | 2 weeks | Phase 1 + P95 <100ms test passing |
| 3 | Model Development | 3 weeks | Phase 2 + 30d historical data |
| 4 | Signal Generation & Filtering | 1 week | Phase 3 |
| 5 | Compounding & Sizing | 1 week | Phase 4 |
| 6 | Cooling Down System | 1 week | Phase 5 |
| 7 | Fast Warm-Up & Boot | 1 week | Phase 6 |
| 8 | Backtesting & Validation | 1 week | Phase 7 |
| 9 | Live Trading Integration | 1 week | Phase 8 passing |
| 10 | Web Dashboard (11 panels) | 2 weeks | Phase 9 (can run parallel) |
| 11 | Monitoring & CI | Ongoing | Phase 9 deployed |
| **Total to Live** | | **~15 weeks** | |

## Appendix B — Go/No-Go Checklist

- [ ] Feature P95 latency < 100ms (automated test passing)
- [ ] No Python for-loops in any hot-path file (lint clean)
- [ ] I/O and CPU separation verified (code review sign-off)
- [ ] All Numba JIT kernels pre-compiled at boot
- [ ] Ring Buffer verified zero-copy reads
- [ ] Backtest win rate >57% on 6+ months OOS
- [ ] Cooling down simulation in backtest verified
- [ ] Paper trading win rate >53% over 50+ bets
- [ ] ≥1 cooling down triggered and recovered in paper mode
- [ ] ≥1 bracket upgrade observed in paper mode
- [ ] Cold boot completes in <3 minutes
- [ ] Warm boot completes in <45 seconds
- [ ] Feature collection verified continuous during cooling down
- [ ] All 6 risk gates tested
- [ ] Hard stop trigger tested
- [ ] All 11 dashboard panels rendering correct data
- [ ] Dashboard WebSocket updates confirmed (5s refresh)
- [ ] Alerting tested: cooling down, bracket change, hard stop, feed down
- [ ] Polymarket API authentication confirmed
- [ ] Manual kill switch tested
- [ ] Runbook documented

---

*This PRD is intended to be consumed by an AI coding agent. Every phase specifies exact file names, class names, method signatures, SQL schemas, and performance targets. Architectural constraints — vectorization rules, I/O/CPU separation, Ring Buffer design, JIT pre-compilation, cooling down state machine, compounding bracket logic, boot sequence timing, and all 11 dashboard panel queries — are specified at implementation level to enable direct code generation without ambiguity.*