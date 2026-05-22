"""
BTC Prediction Bot — Main Entry Point

Usage:
  Paper mode (default):
    python -m src.main

  Live mode:
    PAPER_MODE=false python -m src.main

  Or set PAPER_MODE=false in your .env file.

What this does:
  1. Loads .env configuration
  2. Initializes TimescaleDB + Redis connections
  3. Warms up Ring Buffer via REST backfill (200 bars)
  4. Runs JIT pre-compile (Numba warm-up)
  5. Starts WebSocket data feeds
  6. Starts Execution Engine (main 5-min trading loop)
  7. Starts Dashboard API in background
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import signal
import time
from datetime import datetime, timezone

import structlog

# Load .env before any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

logger = structlog.get_logger(__name__)

PAPER_MODE: bool = os.getenv("PAPER_MODE", "true").lower() not in ("false", "0", "no")
STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "10.00"))
CANDLE_INTERVAL_SEC = 300  # 5 minutes


async def main() -> None:
    logger.info(
        "btc_bot_starting",
        paper_mode=PAPER_MODE,
        starting_capital=STARTING_CAPITAL,
        mode="PAPER TRADING" if PAPER_MODE else "🚨 LIVE TRADING",
    )

    if not PAPER_MODE:
        _warn_live_mode()

    # ── Graceful shutdown handler ─────────────────────────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows doesn't support loop.add_signal_handler for SIGTERM
            pass

    # ── Component initialization ──────────────────────────────────────────
    try:
        components = await _initialize_components()
    except Exception as e:
        logger.error("initialization_failed", error=str(e))
        sys.exit(1)

    # ── Run until stopped ─────────────────────────────────────────────────
    logger.info("bot_running", paper_mode=PAPER_MODE)
    print("\n" + "=" * 60)
    print(f"  BTC Prediction Bot - {'PAPER MODE' if PAPER_MODE else 'LIVE MODE'}")
    print(f"  Starting Capital: ${STARTING_CAPITAL:.2f}")
    print(f"  Dashboard: http://localhost:{os.getenv('DASHBOARD_API_PORT', '8000')}/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        engine = components.get("engine")
        if engine:
            engine_task = asyncio.create_task(engine.start())
            await asyncio.wait(
                [engine_task, asyncio.create_task(stop_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not engine_task.done():
                await engine.stop()
        else:
            # Run the lightweight prediction loop
            logger.info("starting_prediction_loop")
            prediction_task = asyncio.create_task(
                _run_prediction_loop(components, stop_event)
            )
            await asyncio.wait(
                [prediction_task, asyncio.create_task(stop_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
    finally:
        await _cleanup(components)
        logger.info("bot_stopped")


async def _run_prediction_loop(components: dict, stop_event: asyncio.Event) -> None:
    """
    Lightweight prediction loop that runs every 5 minutes.

    Ingests real BTC data from Binance, computes features,
    runs model inference, and when signals fire, executes paper trades
    with full bet logging and balance tracking.
    """
    db_pool = components.get("db_pool")
    ring_buffer = components["ring_buffer"]
    state_manager = components["state_manager"]
    loss_tracker = components["loss_tracker"]
    compounding = components["compounding"]

    # ── Import modules ──
    from src.ingestion.binance_rest import BinanceRESTClient
    from src.features.feature_engine import FeatureEngine
    from src.models.ensemble import EnsembleModel
    from src.signal.signal_generator import SignalGenerator, BotState, Signal
    from src.execution.polymarket_client import PolymarketClient, Market
    from src.execution.paper_trader import PaperTrader
    from src.execution.bet_logger import BetLogger

    rest_client = BinanceRESTClient()  # public endpoints, no API key needed
    feature_engine = FeatureEngine(ring_buffer)
    signal_gen = SignalGenerator()
    paper_trader = PaperTrader(seed=42)
    poly_client = PolymarketClient(paper_mode=True)
    bet_logger = BetLogger(paper_mode=True)

    # ── Try to load trained models ──
    model_dir = "models/latest"
    try:
        import os
        if os.path.exists(os.path.join(model_dir, "ensemble_metadata.json")):
            model = EnsembleModel.load(model_dir)
            logger.info("model_loaded_from_disk", path=model_dir, fitted=model.is_fitted)
        else:
            model = EnsembleModel()  # untrained -> uses heuristic fallback
            logger.info("model_no_weights_found", path=model_dir, note="using heuristic fallback")
    except Exception as e:
        model = EnsembleModel()
        logger.warning("model_load_failed", error=str(e), note="using heuristic fallback")

    # ── Step 1: Backfill ring buffer ──
    logger.info("backfill_starting")
    try:
        candles = await rest_client.fetch_klines("BTCUSDT", "1m", 200)
        for bar in candles:
            ring_buffer.push(bar)
        logger.info("backfill_done", bars=len(candles), buffer_size=ring_buffer.size)
    except Exception as e:
        logger.warning("backfill_failed", error=str(e))

    # Transition to ACTIVE
    state_manager.set_state(BotState.ACTIVE, reason='backfill_complete')
    logger.info("bot_state_active")

    # ── Step 2: Record session in DB ──
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO bot_session_log (started_at, boot_type)
                    VALUES (NOW(), 'cold');
                """)
        except Exception as e:
            logger.warning("session_log_failed", error=str(e))

    # ── Step 3: Main prediction cycle ──
    cycle = 0
    pending_bet: dict | None = None  # Holds the unresolved bet until next candle

    while not stop_event.is_set():
        # Wait for next 5-min candle close
        now = time.time()
        next_close = math.ceil(now / CANDLE_INTERVAL_SEC) * CANDLE_INTERVAL_SEC
        wait_sec = next_close - now
        logger.info("waiting_for_candle", seconds=round(wait_sec, 1),
                     next_close=time.strftime('%H:%M:%S', time.localtime(next_close)))

        # Sleep interruptibly
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_sec)
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass  # candle close time reached

        cycle += 1

        # ── Check cooling down expiry ──
        was_cooling_down = state_manager.is_cooling_down
        if loss_tracker.check_expiry() and db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE cooling_down_log 
                        SET ended_at = NOW(), duration_sec = EXTRACT(EPOCH FROM (NOW() - started_at))
                        WHERE ended_at IS NULL;
                    """)
            except Exception as e:
                logger.warning("cd_log_update_failed", error=str(e))

        # ── Fetch latest candle ──
        try:
            latest = await rest_client.fetch_klines("BTCUSDT", "1m", 5)
            for bar in latest:
                ring_buffer.push(bar)
        except Exception as e:
            logger.warning("candle_fetch_failed", cycle=cycle, error=str(e))
            continue

        # ── Compute features ──
        try:
            features = await feature_engine.compute_all(state_manager.state.name)
        except Exception as e:
            logger.warning("feature_compute_failed", cycle=cycle, error=str(e))
            continue

        if features.get('_insufficient_data'):
            logger.info("insufficient_data_skipping", cycle=cycle)
            continue

        # Get current BTC price from ring buffer
        latest_bar = ring_buffer.get_latest()
        btc_price = latest_bar.get('close', 0)

        # ── Resolve PREVIOUS cycle's pending bet using this candle's price ──
        if pending_bet is not None:
            prev_receipt = pending_bet['receipt']
            prev_prob    = pending_bet['prob']
            prev_market  = pending_bet['market']
            prev_bet_size = pending_bet['bet_size']
            prev_signal_name = pending_bet['signal_name']

            # exit_price = close of the candle that just completed
            # resolve() only determines win/loss; balance is managed by compounding.
            won, _ = paper_trader.resolve(
                prev_receipt,
                prob=prev_prob,
                current_balance=compounding.balance,
                exit_price=btc_price,
            )

            # Compounding is the single source of truth for balance.
            from src.backtest.cost_model import net_payout
            balance_before = compounding.balance
            pnl = net_payout(prev_bet_size, prev_market.best_price, won)
            loss_tracker.on_bet_resolved(won)
            
            # ── Check if cooling down was triggered ──
            if not was_cooling_down and state_manager.is_cooling_down:
                if db_pool:
                    try:
                        async with db_pool.acquire() as conn:
                            await conn.execute("""
                                INSERT INTO cooling_down_log (started_at)
                                VALUES (NOW());
                            """)
                    except Exception as e:
                        logger.warning("cd_log_insert_failed", error=str(e))
            compounding.on_bet_resolved(
                won=won,
                bet_amount=prev_bet_size,
                payout=prev_bet_size + pnl if won else 0.0,
            )
            new_balance = compounding.balance

            logger.info(
                "paper_trade_executed",
                cycle=cycle,
                signal=prev_signal_name,
                bet_size=round(prev_bet_size, 2),
                won=won,
                pnl=round(pnl, 4),
                new_balance=round(compounding.balance, 2),
                btc_price=round(btc_price, 2),
            )

            # ── Write to bet_log ──
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO bet_log
                                (timestamp, market_id, signal, confidence_tier,
                                 model_prob, bet_size, odds, outcome, pnl,
                                 balance_before, balance_after, bracket_index,
                                 paper_mode)
                            VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, true);
                        """,
                        prev_market.market_id,
                        prev_signal_name,
                        pending_bet['confidence_tier'],
                        float(prev_prob),
                        float(prev_bet_size),
                        float(prev_market.best_price),
                        won,
                        float(pnl),
                        float(balance_before),      # balance_before (snapshot pre-update)
                        float(new_balance),         # balance_after
                        compounding.current_bracket.index,
                        )
                except Exception as e:
                    logger.warning("bet_log_write_failed", cycle=cycle, error=str(e))

            # ── Write bet_details (per-model breakdown) ──
            detail = pending_bet.get('prediction_detail')
            if db_pool and detail:
                try:
                    import json as _json
                    async with db_pool.acquire() as conn:
                        # Get the bet_id we just inserted
                        row = await conn.fetchrow("""
                            SELECT id FROM bet_log ORDER BY timestamp DESC LIMIT 1;
                        """)
                        if row:
                            await conn.execute("""
                                INSERT INTO bet_details
                                    (bet_id, prob_lgbm, prob_xgb, prob_lstm, prob_ensemble,
                                     features_pro, features_contra)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                ON CONFLICT (bet_id) DO NOTHING;
                            """,
                            row['id'],
                            float(detail['prob_lgbm']) if detail.get('prob_lgbm') is not None else None,
                            float(detail['prob_xgb']) if detail.get('prob_xgb') is not None else None,
                            float(detail['prob_lstm']) if detail.get('prob_lstm') is not None else None,
                            float(detail['prob']),
                            _json.dumps(detail.get('top_features_pro', [])),
                            _json.dumps(detail.get('top_features_contra', [])),
                            )
                except Exception as e:
                    logger.warning("bet_details_write_failed", cycle=cycle, error=str(e))

            # ── Update portfolio_state ──
            if db_pool:
                try:
                    async with db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE portfolio_state SET
                                balance = $1,
                                bracket_index = $2,
                                total_bets = total_bets + 1,
                                total_wins = total_wins + $3,
                                consecutive_losses = $4,
                                updated_at = NOW()
                            WHERE id = 1;
                        """,
                        float(compounding.balance),
                        compounding.current_bracket.index,
                        1 if won else 0,
                        loss_tracker.consecutive_losses,
                        )
                except Exception as e:
                    logger.warning("portfolio_update_failed", cycle=cycle, error=str(e))

            pending_bet = None  # Clear pending bet after resolution

        # ── Model inference ──
        prediction_detail = None
        try:
            numeric_features = {k: v for k, v in features.items()
                                if isinstance(v, (int, float)) and k not in
                                ('timestamp', 'buffer_size', '_insufficient_data')}
            feature_names = sorted(numeric_features.keys())

            if model.is_fitted:
                prediction_detail = model.predict_single_detailed(numeric_features, feature_names)
                prob = prediction_detail['prob']
            else:
                prob = _heuristic_prob(numeric_features)
                # Build heuristic detail with actual feature contributions
                prediction_detail = _heuristic_detail(numeric_features, prob)
        except Exception as e:
            logger.warning("model_inference_failed", cycle=cycle, error=str(e))
            prob = 0.5

        # ── Signal generation ──
        sig_result = signal_gen.generate_signal(
            prob=prob,
            filters_ok=True,
            bot_state=state_manager.state,
        )

        signal_name = sig_result.signal.name
        confidence_tier = 'HIGH_CONF' if sig_result.signal.is_high_conf else 'NORMAL'
        is_bet = sig_result.signal.is_bet

        logger.info(
            "prediction_cycle_complete",
            cycle=cycle,
            prob=round(prob, 4),
            signal=signal_name,
            confidence_tier=confidence_tier,
            buffer_size=ring_buffer.size,
            btc_price=round(btc_price, 2),
            is_bet=is_bet,
            balance=round(compounding.balance, 2),
        )

        # ── Write to model_predictions table ──
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO model_predictions
                            (timestamp, model_prob, signal, confidence_tier, bot_state, executed)
                        VALUES (NOW(), $1, $2, $3, $4, $5)
                        ON CONFLICT (timestamp) DO UPDATE SET
                            model_prob = EXCLUDED.model_prob,
                            signal = EXCLUDED.signal,
                            confidence_tier = EXCLUDED.confidence_tier,
                            bot_state = EXCLUDED.bot_state,
                            executed = EXCLUDED.executed;
                    """, float(prob), signal_name, confidence_tier,
                    state_manager.state.name, is_bet)
            except Exception as e:
                logger.warning("prediction_write_failed", cycle=cycle, error=str(e))

        # ── Place paper trade (deferred resolution — resolves next cycle) ──
        if is_bet and compounding.can_bet and pending_bet is None:
            try:
                # Get stub market
                markets = await poly_client.get_active_btc_markets()
                market = markets[0] if markets else None

                if market:
                    bet_size = compounding.resolve_bet_size(
                        sig_result.signal,
                        prob=prob,
                        odds=market.best_price
                    )

                    if bet_size > 0:
                        # Place paper bet, recording entry BTC price
                        receipt = paper_trader.simulate(
                            market, sig_result.signal, bet_size,
                            prob=prob, entry_price=btc_price
                        )

                        # Store bet — will be resolved next cycle
                        pending_bet = {
                            'receipt': receipt,
                            'prob': prob,
                            'market': market,
                            'bet_size': bet_size,
                            'signal_name': signal_name,
                            'confidence_tier': confidence_tier,
                            'prediction_detail': prediction_detail,
                        }
                        logger.info(
                            "paper_bet_pending",
                            cycle=cycle,
                            signal=signal_name,
                            bet_size=round(bet_size, 2),
                            entry_price=round(btc_price, 2),
                        )

            except Exception as e:
                logger.error("trade_execution_failed", cycle=cycle, error=str(e))

    # Cleanup
    await rest_client.close()
    feature_engine.shutdown()


async def _initialize_components() -> dict:
    """
    Initialize all subsystems.

    Returns a dict of component name → instance.
    Components that can't connect gracefully degrade (stub mode).
    """
    components: dict = {}

    # 1. Database pool (optional — stub mode if unavailable)
    db_pool = await _init_db()
    components["db_pool"] = db_pool

    # 2. Ring Buffer
    from src.memory.ring_buffer import RingBuffer
    ring_buffer = RingBuffer()
    components["ring_buffer"] = ring_buffer
    logger.info("ring_buffer_ready", size=ring_buffer.size)

    # 3. State Manager + Loss Tracker + Compounding Engine
    from src.signal.signal_generator import BotState
    from src.execution.state_manager import StateManager
    from src.execution.loss_tracker import LossTracker
    from src.sizing.compounding_engine import CompoundingEngine

    initial_balance = STARTING_CAPITAL
    initial_losses = 0

    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT balance, consecutive_losses FROM portfolio_state WHERE id = 1;")
                if row:
                    initial_balance = float(row['balance'])
                    initial_losses = int(row['consecutive_losses'])
                    logger.info("loaded_portfolio_state", balance=initial_balance, consecutive_losses=initial_losses)
                
                # Clear any hanging cooling down logs from a previous aborted run
                await conn.execute("""
                    UPDATE cooling_down_log 
                    SET ended_at = NOW(), duration_sec = EXTRACT(EPOCH FROM (NOW() - started_at))
                    WHERE ended_at IS NULL;
                """)
        except Exception as e:
            logger.warning("failed_to_load_portfolio_state", error=str(e))

    state_manager = StateManager(initial_state=BotState.BOOT)
    loss_tracker = LossTracker(state_manager=state_manager)
    if initial_losses > 0:
        loss_tracker._consecutive_losses = initial_losses
        
    compounding = CompoundingEngine(initial_balance=initial_balance)
    components["state_manager"] = state_manager
    components["loss_tracker"] = loss_tracker
    components["compounding"] = compounding

    # 4. Alert Manager (Telegram if configured, else log only)
    from src.monitoring.alerting import AlertManager, TelegramChannel, EmailChannel
    telegram = TelegramChannel() if os.getenv("TELEGRAM_BOT_TOKEN") else None
    alert_manager = AlertManager(telegram=telegram)
    components["alert_manager"] = alert_manager

    # 5. Performance tracker + drift detector
    from src.monitoring.performance_tracker import PerformanceTracker
    perf_tracker = PerformanceTracker()
    components["perf_tracker"] = perf_tracker

    logger.info("components_initialized", paper_mode=PAPER_MODE)
    return components


async def _init_db():
    """Try to connect asyncpg pool. Return None if DB is not available."""
    try:
        import asyncpg
        host = os.getenv("TIMESCALE_HOST", "localhost")
        port = int(os.getenv("TIMESCALE_PORT", "5432"))
        db = os.getenv("TIMESCALE_DB", "btc_prediction")
        user = os.getenv("TIMESCALE_USER", "btcbot")
        password = os.getenv("TIMESCALE_PASSWORD", "btcbot_secret")

        pool = await asyncio.wait_for(
            asyncpg.create_pool(
                host=host, port=port, database=db, user=user, password=password,
                min_size=2, max_size=10,
            ),
            timeout=5.0,
        )
        logger.info("database_connected", host=host, db=db)
        return pool
    except Exception as e:
        logger.warning("database_unavailable_running_without_db", reason=str(e))
        return None


async def _cleanup(components: dict) -> None:
    """Close all connections gracefully."""
    pool = components.get("db_pool")
    if pool:
        try:
            await pool.close()
        except Exception:
            pass


def _warn_live_mode() -> None:
    """Print a prominent live mode warning."""
    print("\n" + "!" * 60)
    print("  *** LIVE TRADING MODE ACTIVE ***")
    print("  Real money will be wagered on Polymarket.")
    print("  Ensure paper trading validation passed first:")
    print("  - >=50 bets placed")
    print("  - >53% win rate achieved")
    print("!" * 60 + "\n")

    # 3-second pause to notice the warning
    import time
    time.sleep(3)


def _heuristic_detail(features: dict, prob: float) -> dict:
    """
    Build a prediction detail dict from the heuristic formula.
    Decomposes the heuristic into per-feature contributions so users
    can see what drove the prediction even without trained ML models.
    """
    import math

    direction_up = prob >= 0.5

    # Extract the same features used by _heuristic_prob
    ret_1 = features.get('return_1m', 0.0)
    ret_5 = features.get('return_5m', 0.0)
    ret_10 = features.get('return_10m', 0.0)
    ret_20 = features.get('return_20m', 0.0)
    taker_buy_ratio = features.get('taker_buy_ratio', 0.5)
    vol_delta_5 = features.get('volume_delta_ratio_5', 0.0)
    vol_5 = features.get('realized_vol_5', 0.001)

    # Compute each feature's individual contribution to the z-score
    # (same weights as in _heuristic_prob)
    contributions = [
        {
            'name': 'return_1m',
            'importance': round(abs(0.15 * (ret_1 / max(vol_5, 1e-6)) * 0.6), 4),
            'value': round(ret_1, 6),
            'score': 0.15 * (ret_1 / max(vol_5, 1e-6)) * 0.6,
        },
        {
            'name': 'return_5m',
            'importance': round(abs(0.30 * (ret_5 / max(vol_5, 1e-6)) * 0.6), 4),
            'value': round(ret_5, 6),
            'score': 0.30 * (ret_5 / max(vol_5, 1e-6)) * 0.6,
        },
        {
            'name': 'return_10m',
            'importance': round(abs(0.25 * (ret_10 / max(vol_5, 1e-6)) * 0.6), 4),
            'value': round(ret_10, 6),
            'score': 0.25 * (ret_10 / max(vol_5, 1e-6)) * 0.6,
        },
        {
            'name': 'return_20m',
            'importance': round(abs(0.15 * (ret_20 / max(vol_5, 1e-6)) * 0.6), 4),
            'value': round(ret_20, 6),
            'score': 0.15 * (ret_20 / max(vol_5, 1e-6)) * 0.6,
        },
        {
            'name': 'taker_buy_ratio',
            'importance': round(abs((taker_buy_ratio - 0.5) * 4.0 * 0.4), 4),
            'value': round(taker_buy_ratio, 6),
            'score': (taker_buy_ratio - 0.5) * 4.0 * 0.4,
        },
        {
            'name': 'volume_delta_ratio_5',
            'importance': round(abs(vol_delta_5 * 2.0 * 0.4), 4),
            'value': round(vol_delta_5, 6),
            'score': vol_delta_5 * 2.0 * 0.4,
        },
        {
            'name': 'realized_vol_5',
            'importance': round(vol_5 * 100, 4),  # scale for visibility
            'value': round(vol_5, 6),
            'score': 0,  # dampening factor, not directional
        },
    ]

    # Sort by importance descending
    contributions.sort(key=lambda x: x['importance'], reverse=True)

    # Classify as pro/contra based on whether the score aligns with predicted direction
    pro = []
    contra = []
    for c in contributions:
        entry = {'name': c['name'], 'importance': c['importance'], 'value': c['value']}
        score = c['score']
        if direction_up:
            is_pro = score >= 0
        else:
            is_pro = score <= 0

        if is_pro:
            pro.append(entry)
        else:
            contra.append(entry)

    return {
        'prob': prob,
        'prob_lgbm': None,
        'prob_xgb': None,
        'prob_lstm': None,
        'top_features_pro': pro[:10],
        'top_features_contra': contra[:10],
    }


def _heuristic_prob(features: dict) -> float:
    """
    Simple momentum-based heuristic for P(UP) when no trained model exists.

    Uses a weighted combination of:
      - Short-term momentum (return_1m, return_5m)
      - Volume delta (taker_buy_ratio if available)
      - Volatility-adjusted signal strength

    Returns a probability in [0.15, 0.85] range.
    This is NOT a production model — it's a bootstrap heuristic that
    generates signals when trained ML models are unavailable.
    """
    import math

    # Price momentum signals
    ret_1 = features.get('return_1m', 0.0)
    ret_5 = features.get('return_5m', 0.0)
    ret_10 = features.get('return_10m', 0.0)
    ret_20 = features.get('return_20m', 0.0)

    # Volume signals
    taker_buy_ratio = features.get('taker_buy_ratio', 0.5)
    vol_delta_5 = features.get('volume_delta_ratio_5', 0.0)

    # Volatility (for signal dampening)
    vol_5 = features.get('realized_vol_5', 0.001)

    # Combine signals with weights
    # Short-term momentum: recent returns scaled by volatility
    momentum_score = (
        0.15 * (ret_1 / max(vol_5, 1e-6)) +
        0.30 * (ret_5 / max(vol_5, 1e-6)) +
        0.25 * (ret_10 / max(vol_5, 1e-6)) +
        0.15 * (ret_20 / max(vol_5, 1e-6))
    )

    # Volume imbalance: taker buy ratio > 0.5 = bullish
    volume_score = (taker_buy_ratio - 0.5) * 4.0  # scale to [-2, 2]
    volume_score += vol_delta_5 * 2.0

    # Combined z-score
    z = momentum_score * 0.6 + volume_score * 0.4

    # Clamp z-score to reasonable range
    z = max(-3.0, min(3.0, z))

    # Sigmoid to convert z-score to probability
    prob = 1.0 / (1.0 + math.exp(-z))

    # Clamp to [0.15, 0.85] to avoid extreme predictions
    prob = max(0.15, min(0.85, prob))

    return prob


if __name__ == "__main__":
    asyncio.run(main())
