"""
Execution Engine — Main 5-minute candle cycle orchestrator.

PRD 12.2 Main Execution Loop:
  1. Wait for 5-minute candle close
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

Architecture:
  - Single asyncio task per cycle (no concurrency within one cycle)
  - Feature computation in thread pool (CPU-bound, non-blocking)
  - All market/API calls are async
  - DB writes are fire-and-forget via persistence queue
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any

import structlog

from src.signal.signal_generator import BotState, Signal, SignalGenerator
from src.execution.state_manager import StateManager
from src.execution.loss_tracker import LossTracker
from src.execution.polymarket_client import PolymarketClient, CircuitOpenError
from src.execution.market_selector import MarketSelector
from src.execution.paper_trader import PaperTrader
from src.execution.bet_logger import BetLogger
from src.sizing.compounding_engine import CompoundingEngine

logger = structlog.get_logger(__name__)

PAPER_MODE = True            # default: paper mode until validation complete
CANDLE_INTERVAL_SEC = 300    # 5 minutes


class ExecutionEngineStatus(Enum):
    STOPPED  = "STOPPED"
    RUNNING  = "RUNNING"
    STOPPING = "STOPPING"


class ExecutionEngine:
    """
    Main execution loop orchestrating the full bet cycle.

    Wires together all subsystems:
      FeatureEngine → SignalGenerator → MarketSelector →
      PaperTrader/PolymarketClient → LossTracker → BetLogger

    Usage:
        engine = ExecutionEngine(...)
        await engine.start()        # runs indefinitely
        await engine.stop()         # graceful shutdown
        await engine.run_one_cycle()  # single cycle for testing
    """

    def __init__(
        self,
        feature_engine: Any,
        model: Any,
        state_manager: StateManager,
        loss_tracker: LossTracker,
        compounding_engine: CompoundingEngine,
        polymarket_client: PolymarketClient,
        market_selector: MarketSelector,
        paper_trader: PaperTrader,
        bet_logger: BetLogger,
        persistence_queue: Optional[asyncio.Queue] = None,
        paper_mode: bool = PAPER_MODE,
        on_alert: Optional[Callable[[str], None]] = None,
    ):
        self._feature_engine = feature_engine
        self._model = model
        self._state_manager = state_manager
        self._loss_tracker = loss_tracker
        self._compounding = compounding_engine
        self._client = polymarket_client
        self._selector = market_selector
        self._paper_trader = paper_trader
        self._bet_logger = bet_logger
        self._persistence_queue = persistence_queue
        self._paper_mode = paper_mode
        self._on_alert = on_alert
        self._signal_gen = SignalGenerator()

        self._status = ExecutionEngineStatus.STOPPED
        self._cycle_count = 0
        self._bet_count = 0
        self._last_cycle_ms: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run indefinitely, one cycle per 5-minute candle close."""
        self._status = ExecutionEngineStatus.RUNNING
        logger.info("execution_engine_started", paper_mode=self._paper_mode)

        while self._status == ExecutionEngineStatus.RUNNING:
            await self._wait_for_candle_close()
            if self._status != ExecutionEngineStatus.RUNNING:
                break
            await self.run_one_cycle()

        self._status = ExecutionEngineStatus.STOPPED
        logger.info("execution_engine_stopped", cycles=self._cycle_count)

    async def stop(self) -> None:
        """Graceful shutdown — complete current cycle then stop."""
        self._status = ExecutionEngineStatus.STOPPING
        logger.info("execution_engine_stopping")

    async def run_one_cycle(self) -> dict:
        """
        Execute a single 5-minute cycle.

        Returns a cycle report dict for logging/testing.
        This is the exact PRD 12.2 execute_cycle() logic.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1
        report = {
            'cycle': self._cycle_count,
            'action': 'none',
            'state': self._state_manager.state.name,
        }

        try:
            # Step 1: Compute features (always — state-agnostic)
            features = await self._feature_engine.compute_all(
                self._state_manager.state.name
            )
            report['feature_count'] = len(features)

            # Step 2: Push features to DB queue (non-blocking)
            if self._persistence_queue is not None:
                self._persistence_queue.put_nowait(('features', features))

            # Step 3: Check cooling down expiry
            self._loss_tracker.check_expiry()

            # Step 4: Gate — removed early exit to allow model inference during cooling down (PRD 9.4)

            # Step 5: Can we bet? (balance sufficient)
            if not self._compounding.can_bet:
                report['action'] = 'skipped_insufficient_balance'
                return report

            # Step 6: Model inference
            prob = float(self._model.predict_proba(features))
            report['prob'] = round(prob, 4)

            # Step 7: Signal generation with risk gates
            sig_result = self._signal_gen.generate_signal(
                prob=prob,
                filters_ok=True,
                bot_state=self._state_manager.state,
            )
            report['signal'] = sig_result.signal.name

            if sig_result.signal == Signal.NO_BET:
                report['action'] = 'no_bet'
                return report

            # Step 8: Market selection
            try:
                market = await self._selector.find_best_market(sig_result.signal)
            except CircuitOpenError as e:
                logger.error("circuit_breaker_open", error=str(e))
                report['action'] = 'circuit_breaker_open'
                if self._on_alert:
                    self._on_alert(f"Circuit breaker open: {e}")
                return report

            if market is None:
                report['action'] = 'no_market'
                return report

            # Step 9: Bet sizing
            bet_size = self._compounding.resolve_bet_size(
                sig_result.signal,
                prob=prob,
                odds=market.best_price
            )
            if bet_size <= 0:
                report['action'] = 'insufficient_balance'
                return report

            # Step 10: Log placed bet
            confidence_tier = (
                'HIGH_CONF'
                if sig_result.signal in (Signal.BET_UP_HIGH_CONF, Signal.BET_DOWN_HIGH_CONF)
                else 'NORMAL'
            )
            log_entry = self._bet_logger.log_placed(
                market_id=market.market_id,
                signal=sig_result.signal.name,
                confidence_tier=confidence_tier,
                model_prob=prob,
                bet_size=bet_size,
                odds=market.best_price,
                balance_before=self._compounding.balance,
                bracket_index=self._compounding.current_bracket.index,
                features=features,
            )

            # Step 11: Execute (paper or live)
            if self._paper_mode:
                receipt = self._paper_trader.simulate(
                    market, sig_result.signal, bet_size, prob=prob
                )
                # resolve() only determines win/loss outcome; balance is NOT
                # taken from here — compounding is the single source of truth.
                won, _ = self._paper_trader.resolve(
                    receipt, prob=prob,
                    current_balance=self._compounding.balance,
                )
            else:
                from src.execution.polymarket_client import BetSide
                side = BetSide.UP if 'UP' in sig_result.signal.name else BetSide.DOWN
                receipt = await self._client.place_bet(
                    market.market_id, side, bet_size, market.best_price
                )
                won = await self._wait_for_resolution(receipt)

            # Step 12: Update systems — compounding is the single source of truth
            # for balance. net_payout (with fees/slippage) is used via cost_model.
            from src.backtest.cost_model import net_payout
            pnl = net_payout(bet_size, market.best_price, won)
            self._loss_tracker.on_bet_resolved(won)
            self._compounding.on_bet_resolved(
                won=won,
                bet_amount=bet_size,
                # Pass the already-fee-adjusted payout so compounding sees
                # the same net number as cost_model. On win: stake + net profit.
                # On loss: 0 (compounding deducts bet_amount internally).
                payout=bet_size + pnl if won else 0.0,
            )
            new_balance = self._compounding.balance
            self._bet_logger.log_resolved(log_entry, won=won, pnl=pnl, new_balance=new_balance)

            self._bet_count += 1
            report['action'] = 'bet_placed'
            report['won'] = won
            report['pnl'] = round(pnl, 4)
            report['new_balance'] = round(new_balance, 4)

        except CircuitOpenError as e:
            report['action'] = 'circuit_breaker_open'
            logger.error("circuit_breaker_open", error=str(e))
        except Exception as e:
            report['action'] = f'error: {e}'
            logger.error("cycle_error", error=str(e))

        self._last_cycle_ms = (time.perf_counter() - t0) * 1000
        report['latency_ms'] = round(self._last_cycle_ms, 1)
        return report

    async def _wait_for_candle_close(self) -> None:
        """Wait until the next 5-minute candle closes."""
        now = time.time()
        # Next 5-min boundary
        next_close = (int(now / CANDLE_INTERVAL_SEC) + 1) * CANDLE_INTERVAL_SEC
        wait = next_close - now
        logger.debug("waiting_for_candle", seconds=round(wait, 1))
        await asyncio.sleep(wait)

    async def _wait_for_resolution(self, receipt) -> bool:
        """Poll for bet resolution (live mode only)."""
        for _ in range(60):  # poll up to 5 minutes
            await asyncio.sleep(5)
            status = await self._client.get_bet_status(receipt.order_id)
            if status.status.name in ('WON', 'LOST'):
                return status.status.name == 'WON'
        return False  # timeout → assume loss

    def stats(self) -> dict:
        return {
            'status': self._status.name,
            'cycle_count': self._cycle_count,
            'bet_count': self._bet_count,
            'paper_mode': self._paper_mode,
            'last_cycle_ms': round(self._last_cycle_ms, 1),
            'state': self._state_manager.state.name,
            'balance': self._compounding.balance,
        }
