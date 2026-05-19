"""
Tests for Phase 8 — Backtesting & Validation.

Covers:
  1. Cost model (net_payout, effective_odds, break_even)
  2. Backtester engine (run, cooling down, bracket progression)
  3. BacktestResult metrics (win_rate, drawdown, entry_rate)
  4. Report generator (streaks, PnL by signal, bracket transitions)
  5. Leakage detector (4 anti-leakage checks)
  6. Integration: full backtest with deterministic outcomes
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.backtest.cost_model import (
    net_payout, effective_odds, break_even_win_rate,
    POLYMARKET_FEE, SLIPPAGE_BPS,
)
from src.backtest.backtester import Backtester, BacktestResult, COOLING_DOWN_BARS
from src.backtest.report_generator import ReportGenerator
from src.backtest.leakage_detector import LeakageDetector


# ═══════════════════════════════════════════════════════════════════════════════
# Cost Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostModel:
    def test_polymarket_fee_constant(self):
        assert POLYMARKET_FEE == 0.02

    def test_slippage_constant(self):
        assert SLIPPAGE_BPS == 5

    def test_win_payout(self):
        pnl = net_payout(1.0, 1.90, won=True)
        # gross = 0.90, fee = 0.018, slippage = 0.0005
        expected = 0.90 - 0.018 - 0.0005
        assert abs(pnl - expected) < 0.001

    def test_loss_payout(self):
        pnl = net_payout(1.0, 1.90, won=False)
        # -(bet + slippage) = -(1.0 + 0.0005)
        assert abs(pnl - (-1.0005)) < 0.001

    def test_win_payout_larger_bet(self):
        pnl = net_payout(2.0, 1.90, won=True)
        gross = 2.0 * 0.90
        fee = gross * 0.02
        slippage = 2.0 * 5 / 10000
        assert abs(pnl - (gross - fee - slippage)) < 0.001

    def test_loss_payout_larger_bet(self):
        pnl = net_payout(2.0, 1.90, won=False)
        slippage = 2.0 * 5 / 10000
        assert abs(pnl - (-(2.0 + slippage))) < 0.001

    def test_effective_odds(self):
        eff = effective_odds(1.90)
        # 0.90 * (1 - 0.02) = 0.882 => eff = 1.882
        assert abs(eff - 1.882) < 0.001

    def test_break_even_win_rate(self):
        be = break_even_win_rate(1.90)
        # 1 / 1.882 ≈ 0.5314
        assert 0.52 < be < 0.55

    def test_win_at_even_odds(self):
        pnl = net_payout(1.0, 2.0, won=True)
        # gross=1.0, fee=0.02, slippage=0.0005
        assert pnl > 0

    def test_loss_always_negative(self):
        for odds in [1.5, 1.9, 2.5, 3.0]:
            assert net_payout(1.0, odds, won=False) < 0


# ═══════════════════════════════════════════════════════════════════════════════
# Backtester Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktester:
    def test_empty_probabilities(self):
        bt = Backtester(initial_capital=10.0)
        result = bt.run(np.array([]))
        assert result.total_bets == 0
        assert result.final_capital == 10.0

    def test_all_low_confidence_no_bets(self):
        """Probabilities below threshold → no bets placed."""
        bt = Backtester(initial_capital=10.0, signal_threshold=0.55)
        probs = np.full(100, 0.50)  # below threshold
        result = bt.run(probs)
        assert result.total_bets == 0

    def test_high_confidence_places_bets(self):
        bt = Backtester(initial_capital=10.0)
        np.random.seed(42)
        probs = np.random.uniform(0.60, 0.85, size=50)
        result = bt.run(probs)
        assert result.total_bets > 0

    def test_win_rate_with_all_wins(self):
        bt = Backtester(initial_capital=10.0, odds=1.90)
        probs = np.full(20, 0.80)
        actuals = np.ones(20, dtype=int)  # all UP, all correct
        result = bt.run(probs, actuals=actuals)
        assert result.total_bets > 0
        assert result.win_rate == 1.0
        assert result.final_capital > 10.0

    def test_win_rate_with_all_losses(self):
        bt = Backtester(initial_capital=10.0, odds=1.90)
        probs = np.full(20, 0.80)
        actuals = np.zeros(20, dtype=int)  # all DOWN, bets are UP → all lose
        result = bt.run(probs, actuals=actuals)
        assert result.total_bets > 0
        assert result.win_rate == 0.0
        assert result.final_capital < 10.0

    def test_cooling_down_triggers(self):
        """2 consecutive losses → cooling down → skip 6 bars."""
        bt = Backtester(initial_capital=10.0)
        # Create pattern: high prob → will bet. All lose → triggers cooling down
        probs = np.full(30, 0.80)
        actuals = np.zeros(30, dtype=int)
        result = bt.run(probs, actuals=actuals)
        assert result.cooling_down_events > 0
        assert result.cooling_down_bars_skipped > 0

    def test_cooling_down_skips_bars(self):
        bt = Backtester(initial_capital=10.0)
        probs = np.full(50, 0.80)
        actuals = np.zeros(50, dtype=int)
        result = bt.run(probs, actuals=actuals)
        # With cooling down, fewer bets should be placed than bars
        assert result.total_bets < 50

    def test_balance_never_negative(self):
        bt = Backtester(initial_capital=10.0)
        np.random.seed(42)
        probs = np.random.uniform(0.55, 0.90, size=200)
        result = bt.run(probs)
        for bet in result.bet_log:
            assert bet.balance >= 0

    def test_drawdown_tracked(self):
        bt = Backtester(initial_capital=10.0)
        probs = np.full(30, 0.75)
        actuals = np.zeros(30, dtype=int)
        result = bt.run(probs, actuals=actuals)
        assert result.max_drawdown_pct > 0

    def test_result_summary_keys(self):
        bt = Backtester(initial_capital=10.0)
        probs = np.full(10, 0.75)
        result = bt.run(probs)
        s = result.summary()
        required = [
            'initial_capital', 'final_capital', 'total_return_pct',
            'total_bets', 'win_rate', 'entry_rate_pct', 'max_drawdown_pct',
            'cooling_down_events',
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_entry_rate(self):
        bt = Backtester(initial_capital=10.0)
        probs = np.full(100, 0.80)
        actuals = np.ones(100, dtype=int)
        result = bt.run(probs, actuals=actuals)
        assert 0 < result.entry_rate <= 100

    def test_bet_log_records(self):
        bt = Backtester(initial_capital=10.0)
        probs = np.full(10, 0.80)
        actuals = np.ones(10, dtype=int)
        result = bt.run(probs, actuals=actuals)
        for bet in result.bet_log:
            assert bet.bet_size > 0
            assert bet.odds > 0
            assert isinstance(bet.won, bool)
            assert isinstance(bet.pnl, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generator Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportGenerator:
    def _run_backtest(self, seed=42):
        bt = Backtester(initial_capital=10.0)
        np.random.seed(seed)
        probs = np.random.uniform(0.55, 0.85, size=100)
        return bt.run(probs)

    def test_report_has_summary(self):
        result = self._run_backtest()
        rg = ReportGenerator()
        report = rg.generate(result)
        assert 'win_rate' in report.summary

    def test_streak_analysis(self):
        result = self._run_backtest()
        rg = ReportGenerator()
        report = rg.generate(result)
        assert report.streaks is not None
        assert report.streaks.max_win_streak >= 0
        assert report.streaks.max_loss_streak >= 0

    def test_pnl_by_signal(self):
        result = self._run_backtest()
        rg = ReportGenerator()
        report = rg.generate(result)
        assert isinstance(report.pnl_by_signal, dict)
        if report.pnl_by_signal:
            for sig, data in report.pnl_by_signal.items():
                assert 'count' in data
                assert 'win_rate' in data
                assert 'total_pnl' in data

    def test_bracket_transitions(self):
        result = self._run_backtest()
        rg = ReportGenerator()
        report = rg.generate(result)
        assert isinstance(report.bracket_transitions, list)

    def test_sharpe_ratio(self):
        result = self._run_backtest()
        rg = ReportGenerator()
        report = rg.generate(result)
        assert 'sharpe_ratio' in report.summary

    def test_empty_backtest_report(self):
        bt = Backtester(initial_capital=10.0)
        result = bt.run(np.array([]))
        rg = ReportGenerator()
        report = rg.generate(result)
        assert report.streaks.max_win_streak == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Leakage Detector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeakageDetector:
    def test_no_data_all_pass(self):
        ld = LeakageDetector()
        report = ld.run_all()
        assert report.all_passed

    def test_clean_data_passes(self):
        np.random.seed(42)
        X = np.random.randn(500, 20)
        y = (np.random.rand(500) > 0.5).astype(int)
        ts = np.arange(500, dtype=float) * 300
        train_idx = np.arange(400)
        val_idx = np.arange(400, 500)

        ld = LeakageDetector()
        report = ld.run_all(X, y, ts, train_idx, val_idx)
        assert report.all_passed

    def test_chronological_split_overlap_fails(self):
        np.random.seed(42)
        ts = np.arange(100, dtype=float) * 300
        train_idx = np.arange(80)
        val_idx = np.arange(50, 100)  # overlaps!

        ld = LeakageDetector()
        check = ld.check_chronological_split(ts, train_idx, val_idx)
        assert not check.passed

    def test_chronological_split_strict_passes(self):
        ts = np.arange(100, dtype=float) * 300
        train_idx = np.arange(70)
        val_idx = np.arange(70, 100)

        ld = LeakageDetector()
        check = ld.check_chronological_split(ts, train_idx, val_idx)
        assert check.passed

    def test_label_leakage_detected(self):
        """Feature perfectly correlated with label → leakage."""
        np.random.seed(42)
        y = (np.random.rand(500) > 0.5).astype(int)
        X = np.column_stack([y.astype(float), np.random.randn(500, 5)])

        ld = LeakageDetector()
        check = ld.check_label_shift(y, X)
        assert not check.passed  # label is feature 0

    def test_no_label_leakage(self):
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (np.random.rand(500) > 0.5).astype(int)

        ld = LeakageDetector()
        check = ld.check_label_shift(y, X)
        assert check.passed

    def test_leakage_report_summary(self):
        ld = LeakageDetector()
        report = ld.run_all()
        s = report.summary()
        assert 'all_passed' in s
        assert 'total_checks' in s
        assert s['total_checks'] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestIntegration:
    def test_full_backtest_with_report_and_leakage(self):
        """End-to-end: backtest → report → leakage check."""
        np.random.seed(42)

        # Simulate 6 months of data (288 bars/day × 180 days)
        n_bars = 288 * 30  # 1 month for speed
        probs = np.random.uniform(0.50, 0.90, size=n_bars)
        timestamps = np.arange(n_bars, dtype=float) * 300

        # Run backtest
        bt = Backtester(initial_capital=10.0)
        result = bt.run(probs, timestamps=timestamps)

        # Generate report
        rg = ReportGenerator()
        report = rg.generate(result)

        # Verify report structure
        assert result.total_bars == n_bars
        assert result.total_bets > 0
        assert 0 <= result.win_rate <= 1.0
        assert result.cooling_down_events >= 0
        assert report.streaks is not None

        # Leakage check on synthetic features
        X = np.random.randn(n_bars, 20)
        y = (np.random.rand(n_bars) > 0.5).astype(int)
        train_idx = np.arange(n_bars * 3 // 4)
        val_idx = np.arange(n_bars * 3 // 4, n_bars)

        ld = LeakageDetector()
        leakage_report = ld.run_all(X, y, timestamps, train_idx, val_idx)
        assert leakage_report.all_passed

    def test_cooling_down_simulation_mirrors_production(self):
        """
        PRD 11.1: Cooling down in backtest must mirror production.
        2 consecutive losses → skip 6 bars (30 min).
        """
        # Force exact pattern: W, L, L, (skip 6), W, L, L, (skip 6)
        bt = Backtester(initial_capital=10.0)

        # All high prob → will bet every bar
        probs = np.full(30, 0.80)
        # Pattern: win, loss, loss, then 6 skipped bars...
        actuals = np.array([
            1, 0, 0,           # W L L → cooling down
            1, 1, 1, 1, 1, 1, # skipped (cooling down)
            1, 0, 0,           # W L L → cooling down
            1, 1, 1, 1, 1, 1, # skipped
            1, 1, 1, 1, 1, 1, # active
            1, 1, 1, 1, 1, 1, # active
        ])

        result = bt.run(probs, actuals=actuals)
        assert result.cooling_down_events >= 1
        assert result.cooling_down_bars_skipped >= COOLING_DOWN_BARS


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
