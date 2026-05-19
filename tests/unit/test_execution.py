"""
Tests for Phase 9 — Live Trading Integration.

Covers:
  1. PolymarketClient: paper mode, circuit breaker (5 failures), retry
  2. MarketSelector: direction filter, volume sort, no valid market
  3. PaperTrader: simulate, resolve, win_rate, validation_report
  4. BetLogger: log_placed, log_resolved, queue push, recent_bets
  5. ExecutionEngine: full cycle (paper mode), state gate, cooling down
  6. Circuit breaker: trips at 5 failures, resets on success
"""

import asyncio
import pytest
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.execution.polymarket_client import (
    PolymarketClient, Market, BetSide, CircuitOpenError,
    CIRCUIT_BREAKER_THRESHOLD,
)
from src.execution.market_selector import MarketSelector, MIN_VOLUME_USD
from src.execution.paper_trader import PaperTrader
from src.execution.bet_logger import BetLogger
from src.execution.execution_engine import ExecutionEngine
from src.execution.state_manager import StateManager
from src.execution.loss_tracker import LossTracker
from src.sizing.compounding_engine import CompoundingEngine
from src.signal.signal_generator import Signal, BotState


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeModel:
    def __init__(self, prob: float = 0.75):
        self._prob = prob

    def predict_proba(self, features) -> float:
        return self._prob


class FakeFeatureEngine:
    async def compute_all(self, bot_state: str) -> dict:
        return {f'f_{i}': float(i) for i in range(20)}


# ═══════════════════════════════════════════════════════════════════════════════
# Polymarket Client Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolymarketClient:
    def test_paper_mode_get_markets(self):
        client = PolymarketClient(paper_mode=True)
        markets = run_async(client.get_active_btc_markets())
        assert isinstance(markets, list)
        assert len(markets) >= 1
        assert markets[0].best_price > 1.0

    def test_paper_mode_get_odds(self):
        client = PolymarketClient(paper_mode=True)
        odds = run_async(client.get_market_odds("btc-001"))
        assert odds > 1.0

    def test_paper_mode_place_bet(self):
        client = PolymarketClient(paper_mode=True)
        receipt = run_async(
            client.place_bet("btc-001", BetSide.UP, 1.0, 1.90)
        )
        assert receipt.paper is True
        assert receipt.order_id.startswith("paper_")
        assert receipt.size == 1.0

    def test_paper_mode_get_status(self):
        client = PolymarketClient(paper_mode=True)
        status = run_async(client.get_bet_status("paper_123"))
        assert status.order_id == "paper_123"

    def test_stats_structure(self):
        client = PolymarketClient(paper_mode=True)
        run_async(client.get_active_btc_markets())
        s = client.stats()
        assert 'total_calls' in s
        assert 'circuit_open' in s
        assert s['paper_mode'] is True


class TestCircuitBreaker:
    def test_circuit_breaker_threshold(self):
        assert CIRCUIT_BREAKER_THRESHOLD == 5

    def test_circuit_open_after_5_failures(self):
        client = PolymarketClient(paper_mode=False)
        # Simulate 5 consecutive failures directly
        client._consecutive_failures = 5
        with pytest.raises(CircuitOpenError):
            run_async(client.get_active_btc_markets())

    def test_circuit_not_open_at_4(self):
        client = PolymarketClient(paper_mode=True)
        client._consecutive_failures = 4
        # Paper mode succeeds even with 4 failures accumulated
        markets = run_async(client.get_active_btc_markets())
        assert isinstance(markets, list)

    def test_circuit_resets_on_success(self):
        client = PolymarketClient(paper_mode=True)
        client._consecutive_failures = 3
        # _on_success is called by _with_retry on successful non-paper calls.
        # Verify the mechanism directly — reset to 0 on success.
        client._on_success()
        assert client._consecutive_failures == 0

    def test_stats_circuit_open_flag(self):
        client = PolymarketClient(paper_mode=True)
        client._consecutive_failures = 5
        s = client.stats()
        assert s['circuit_open'] is True

    def test_stats_circuit_closed_flag(self):
        client = PolymarketClient(paper_mode=True)
        s = client.stats()
        assert s['circuit_open'] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Market Selector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketSelector:
    def _make_selector(self, markets=None):
        client = PolymarketClient(paper_mode=True)
        if markets is not None:
            client._stub = lambda: markets
            # Override the stub
            async def custom_markets():
                return markets
            client.get_active_btc_markets = custom_markets
        return MarketSelector(client)

    def test_finds_market_for_up_signal(self):
        sel = self._make_selector()
        market = run_async(sel.find_best_market(Signal.BET_UP_NORMAL))
        assert market is not None
        assert market.best_price > 1.0

    def test_no_bet_returns_none(self):
        sel = self._make_selector()
        market = run_async(sel.find_best_market(Signal.NO_BET))
        assert market is None

    def test_no_valid_market_returns_none(self):
        low_vol = [Market("m1", "Q?", "YES", 1.90, volume_24h=10.0)]
        sel = self._make_selector(markets=low_vol)
        market = run_async(sel.find_best_market(Signal.BET_UP_NORMAL))
        assert market is None

    def test_selects_highest_volume(self):
        markets = [
            Market("m1", "Q1", "YES", 1.90, volume_24h=5_000.0),
            Market("m2", "Q2", "YES", 1.85, volume_24h=50_000.0),
        ]
        sel = self._make_selector(markets=markets)
        market = run_async(sel.find_best_market(Signal.BET_UP_NORMAL))
        assert market.market_id == "m2"

    def test_min_volume_filter(self):
        assert MIN_VOLUME_USD == 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Trader Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperTrader:
    def _make_market(self):
        return Market("btc-001", "BTC up?", "YES", 1.90, 10_000.0)

    def test_simulate_returns_receipt(self):
        pt = PaperTrader(seed=42)
        receipt = pt.simulate(self._make_market(), Signal.BET_UP_NORMAL, 1.0, prob=0.75)
        assert receipt.paper is True
        assert receipt.order_id.startswith("paper_")
        assert receipt.size == 1.0

    def test_resolve_returns_bool_and_balance(self):
        pt = PaperTrader(seed=42)
        receipt = pt.simulate(self._make_market(), Signal.BET_UP_NORMAL, 1.0, prob=0.75)
        won, balance = pt.resolve(receipt, prob=0.75, current_balance=10.0)
        assert isinstance(won, bool)
        assert isinstance(balance, float)

    def test_high_prob_mostly_wins(self):
        pt = PaperTrader(seed=42)
        market = self._make_market()
        wins = 0
        for _ in range(100):
            r = pt.simulate(market, Signal.BET_UP_NORMAL, 1.0, prob=0.95)
            won, _ = pt.resolve(r, prob=0.95, current_balance=10.0)
            wins += int(won)
        assert wins > 70  # ~95% expected

    def test_win_rate_property(self):
        pt = PaperTrader(seed=42)
        market = self._make_market()
        for _ in range(20):
            r = pt.simulate(market, Signal.BET_UP_NORMAL, 1.0, prob=0.70)
            pt.resolve(r, prob=0.70, current_balance=10.0)
        assert 0.0 <= pt.win_rate <= 1.0

    def test_validation_report_structure(self):
        pt = PaperTrader(seed=42)
        report = pt.validation_report()
        assert 'total_bets' in report
        assert 'win_rate' in report
        assert 'go' in report
        assert 'criteria' in report

    def test_50_bet_go_criteria(self):
        pt = PaperTrader(seed=42)
        market = self._make_market()
        for _ in range(60):
            r = pt.simulate(market, Signal.BET_UP_NORMAL, 1.0, prob=0.80)
            pt.resolve(r, prob=0.80, current_balance=10.0)
        report = pt.validation_report()
        assert report['criteria']['min_50_bets'] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Bet Logger Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBetLogger:
    def test_log_placed_creates_entry(self):
        bl = BetLogger(paper_mode=True)
        entry = bl.log_placed(
            market_id="btc-001", signal="BET_UP_NORMAL",
            confidence_tier="NORMAL", model_prob=0.75,
            bet_size=1.0, odds=1.90, balance_before=10.0,
            bracket_index=0,
        )
        assert entry.market_id == "btc-001"
        assert entry.outcome is None  # pending

    def test_log_resolved_updates_entry(self):
        bl = BetLogger(paper_mode=True)
        entry = bl.log_placed(
            market_id="btc-001", signal="BET_UP_NORMAL",
            confidence_tier="NORMAL", model_prob=0.75,
            bet_size=1.0, odds=1.90, balance_before=10.0,
            bracket_index=0,
        )
        bl.log_resolved(entry, won=True, pnl=0.88, new_balance=10.88)
        assert entry.outcome is True
        assert entry.pnl == 0.88
        assert entry.balance_after == 10.88

    def test_queue_push_on_resolve(self):
        queue = asyncio.Queue()
        bl = BetLogger(persistence_queue=queue, paper_mode=True)
        entry = bl.log_placed(
            market_id="m1", signal="BET_UP_NORMAL",
            confidence_tier="NORMAL", model_prob=0.75,
            bet_size=1.0, odds=1.90, balance_before=10.0,
            bracket_index=0,
        )
        bl.log_resolved(entry, won=True, pnl=0.88, new_balance=10.88)
        assert queue.qsize() == 1

    def test_recent_bets_buffer(self):
        bl = BetLogger(paper_mode=True)
        for i in range(5):
            bl.log_placed(
                market_id=f"m{i}", signal="BET_UP_NORMAL",
                confidence_tier="NORMAL", model_prob=0.7,
                bet_size=1.0, odds=1.9, balance_before=10.0,
                bracket_index=0,
            )
        assert len(bl.recent_bets) == 5

    def test_to_db_row(self):
        bl = BetLogger(paper_mode=True)
        entry = bl.log_placed(
            market_id="m1", signal="BET_UP_NORMAL",
            confidence_tier="NORMAL", model_prob=0.75,
            bet_size=1.0, odds=1.90, balance_before=10.0,
            bracket_index=0, features={'f1': 1.0},
        )
        row = entry.to_db_row()
        assert 'market_id' in row
        assert 'feature_snapshot' in row


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionEngine:
    def _make_engine(self, prob=0.75, state=BotState.ACTIVE, balance=10.0):
        sm = StateManager(initial_state=state)
        lt = LossTracker(sm)
        ce = CompoundingEngine(balance)
        client = PolymarketClient(paper_mode=True)
        sel = MarketSelector(client)
        pt = PaperTrader(seed=42)
        bl = BetLogger(paper_mode=True)
        model = FakeModel(prob=prob)
        fe = FakeFeatureEngine()

        return ExecutionEngine(
            feature_engine=fe,
            model=model,
            state_manager=sm,
            loss_tracker=lt,
            compounding_engine=ce,
            polymarket_client=client,
            market_selector=sel,
            paper_trader=pt,
            bet_logger=bl,
            paper_mode=True,
        ), sm, lt, ce

    def test_cycle_when_active_high_prob(self):
        engine, sm, lt, ce = self._make_engine(prob=0.80)
        report = run_async(engine.run_one_cycle())
        assert report['action'] in ('bet_placed', 'no_bet', 'no_market')

    def test_cycle_skipped_when_cooling_down(self):
        sm = StateManager(initial_state=BotState.ACTIVE, cooling_down_duration=100.0)
        sm.set_state(BotState.COOLING_DOWN, reason="test")
        lt = LossTracker(sm)
        ce = CompoundingEngine(10.0)
        client = PolymarketClient(paper_mode=True)
        sel = MarketSelector(client)
        pt = PaperTrader(seed=42)
        bl = BetLogger(paper_mode=True)
        engine = ExecutionEngine(
            feature_engine=FakeFeatureEngine(),
            model=FakeModel(0.80),
            state_manager=sm,
            loss_tracker=lt,
            compounding_engine=ce,
            polymarket_client=client,
            market_selector=sel,
            paper_trader=pt,
            bet_logger=bl,
            paper_mode=True,
        )
        report = run_async(engine.run_one_cycle())
        assert report['action'] == 'no_bet'

    def test_cycle_no_bet_low_prob(self):
        engine, sm, lt, ce = self._make_engine(prob=0.45)
        report = run_async(engine.run_one_cycle())
        assert report['action'] == 'no_bet'

    def test_cycle_has_latency(self):
        engine, _, _, _ = self._make_engine(prob=0.80)
        report = run_async(engine.run_one_cycle())
        assert 'latency_ms' in report
        assert report['latency_ms'] < 500  # PRD 12.3 latency < 500ms

    def test_stats_structure(self):
        engine, _, _, _ = self._make_engine()
        run_async(engine.run_one_cycle())
        s = engine.stats()
        assert 'cycle_count' in s
        assert 'bet_count' in s
        assert s['paper_mode'] is True

    def test_multiple_cycles_accumulate(self):
        engine, _, _, _ = self._make_engine(prob=0.80)
        for _ in range(5):
            run_async(engine.run_one_cycle())
        assert engine.stats()['cycle_count'] == 5

    def test_insufficient_balance_skips(self):
        engine, _, _, _ = self._make_engine(balance=0.10)
        report = run_async(engine.run_one_cycle())
        assert report['action'] in ('insufficient_balance', 'no_bet',
                                    'skipped_insufficient_balance')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
