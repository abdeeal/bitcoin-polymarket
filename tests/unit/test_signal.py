"""
Tests for Phase 4 — Signal Generation & Filtering.

Covers:
  1. Threshold classification (all tiers, boundary values)
  2. HIGH_CONF consecutive signals all fire (no cap)
  3. All 6 gates pass/fail independently
  4. Gate failure does not short-circuit other gates
  5. Odds EV computation (UP/DOWN, fail-open)
  6. Bot state gate blocks all signals during COOLING_DOWN
  7. Full pipeline integration (gates → signal → EV)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.signal.signal_generator import Signal, BotState, SignalGenerator, SignalResult
from src.signal.risk_gates import (
    RiskGates, GateResult, GateThresholds,
    _gate_volatility, _gate_spread, _gate_squeeze,
    _gate_liquidation, _gate_odds_ev, _gate_bot_state,
)
from src.signal.odds_checker import OddsChecker, OddsData


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Generator — Threshold Logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholdClassification:
    """Test threshold boundaries and tiering."""

    def setup_method(self):
        self.gen = SignalGenerator()

    def test_high_conf_up(self):
        result = self.gen.generate_signal(0.85, filters_ok=True)
        assert result.signal == Signal.BET_UP_HIGH_CONF

    def test_high_conf_up_exact_boundary(self):
        result = self.gen.generate_signal(0.82, filters_ok=True)
        assert result.signal == Signal.BET_UP_HIGH_CONF

    def test_normal_up(self):
        result = self.gen.generate_signal(0.75, filters_ok=True)
        assert result.signal == Signal.BET_UP_NORMAL

    def test_normal_up_exact_boundary(self):
        result = self.gen.generate_signal(0.72, filters_ok=True)
        assert result.signal == Signal.BET_UP_NORMAL

    def test_no_bet_uncertainty_zone(self):
        """Values between 0.28 and 0.72 → NO_BET."""
        for prob in [0.29, 0.40, 0.50, 0.60, 0.71]:
            result = self.gen.generate_signal(prob, filters_ok=True)
            assert result.signal == Signal.NO_BET, f"P={prob} should be NO_BET"

    def test_normal_down(self):
        result = self.gen.generate_signal(0.25, filters_ok=True)
        assert result.signal == Signal.BET_DOWN_NORMAL

    def test_normal_down_exact_boundary(self):
        result = self.gen.generate_signal(0.28, filters_ok=True)
        assert result.signal == Signal.BET_DOWN_NORMAL

    def test_high_conf_down(self):
        result = self.gen.generate_signal(0.15, filters_ok=True)
        assert result.signal == Signal.BET_DOWN_HIGH_CONF

    def test_high_conf_down_exact_boundary(self):
        result = self.gen.generate_signal(0.18, filters_ok=True)
        assert result.signal == Signal.BET_DOWN_HIGH_CONF

    def test_high_conf_takes_priority_over_normal(self):
        """A prob of 0.85 must produce HIGH_CONF, not NORMAL (even though 0.85 > 0.72)."""
        result = self.gen.generate_signal(0.85, filters_ok=True)
        assert result.signal == Signal.BET_UP_HIGH_CONF
        assert result.signal != Signal.BET_UP_NORMAL

    def test_filters_false_blocks_high_conf(self):
        """Even a 0.99 probability is blocked when gates fail."""
        result = self.gen.generate_signal(0.99, filters_ok=False)
        assert result.signal == Signal.NO_BET

    def test_filters_false_gates_passed_false(self):
        result = self.gen.generate_signal(0.85, filters_ok=False)
        assert result.gates_passed is False


class TestConsecutiveSignals:
    """
    PRD requirement: HIGH_CONF signals fire consecutively without cap.
    No frequency limiting on any tier.
    """

    def test_consecutive_high_conf_all_fire(self):
        """5 consecutive HIGH_CONF signals must ALL generate BET_UP_HIGH_CONF."""
        gen = SignalGenerator()
        signals = []
        for _ in range(5):
            result = gen.generate_signal(0.85, filters_ok=True)
            signals.append(result.signal)

        assert all(s == Signal.BET_UP_HIGH_CONF for s in signals), \
            "All consecutive HIGH_CONF must fire"

    def test_consecutive_normal_all_fire(self):
        """No cap on NORMAL tier either."""
        gen = SignalGenerator()
        signals = [gen.generate_signal(0.75, filters_ok=True).signal for _ in range(10)]
        assert all(s == Signal.BET_UP_NORMAL for s in signals)

    def test_mixed_consecutive_fire(self):
        """Alternating HIGH_CONF and NORMAL all fire."""
        gen = SignalGenerator()
        probs = [0.85, 0.75, 0.83, 0.74, 0.90]
        results = [gen.generate_signal(p, filters_ok=True) for p in probs]
        assert all(r.signal.is_bet for r in results)

    def test_stats_tracks_count(self):
        gen = SignalGenerator()
        for _ in range(7):
            gen.generate_signal(0.85, filters_ok=True)
        assert gen.stats()['total_signals'] == 7


class TestBotStateBlocking:
    """Bot state gate blocks signals regardless of probability."""

    def test_cooling_down_blocks_all_signals(self):
        gen = SignalGenerator()
        probs = [0.99, 0.85, 0.75, 0.25, 0.15, 0.01]
        for prob in probs:
            result = gen.generate_signal(
                prob, filters_ok=True, bot_state=BotState.COOLING_DOWN
            )
            assert result.signal == Signal.NO_BET, \
                f"COOLING_DOWN must block prob={prob}"

    def test_active_allows_signals(self):
        gen = SignalGenerator()
        result = gen.generate_signal(0.85, filters_ok=True, bot_state=BotState.ACTIVE)
        assert result.signal == Signal.BET_UP_HIGH_CONF

    def test_paused_blocks_signals(self):
        gen = SignalGenerator()
        result = gen.generate_signal(0.85, filters_ok=True, bot_state=BotState.PAUSED)
        assert result.signal == Signal.NO_BET


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Gates — Individual Gate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGate1Volatility:
    """Gate 1: Volatility Regime."""

    def test_normal_vol_passes(self):
        f = {'vol_regime_zscore': 1.0, 'atr_ratio': 1.0}
        passed, reason, _ = _gate_volatility(f, GateThresholds())
        assert passed

    def test_extreme_vol_fails(self):
        f = {'vol_regime_zscore': 2.5, 'atr_ratio': 1.0}
        passed, reason, _ = _gate_volatility(f, GateThresholds())
        assert not passed
        assert 'VOLATILITY' in reason

    def test_exact_boundary_fails(self):
        """vol_regime == 2.0 is at the limit — must fail (< not <=)."""
        f = {'vol_regime_zscore': 2.0, 'atr_ratio': 1.0}
        passed, _, _ = _gate_volatility(f, GateThresholds())
        assert not passed

    def test_extreme_atr_fails(self):
        f = {'vol_regime_zscore': 0.5, 'atr_ratio': 1.8}
        passed, reason, _ = _gate_volatility(f, GateThresholds())
        assert not passed

    def test_missing_features_defaults_to_pass(self):
        """Missing features should not cause exceptions."""
        passed, _, _ = _gate_volatility({}, GateThresholds())
        assert passed  # defaults: vol=0.0, atr=1.0


class TestGate2Spread:
    """Gate 2: Spread filter."""

    def test_normal_spread_passes(self):
        f = {'spread_bps': 2.0, 'spread_mean_20': 2.0}
        passed, reason, _, _ = _gate_spread(f, GateThresholds())
        assert passed

    def test_wide_spread_fails(self):
        f = {'spread_bps': 6.0, 'spread_mean_20': 2.0}  # 3x avg > 2.5
        passed, reason, _, _ = _gate_spread(f, GateThresholds())
        assert not passed
        assert 'SPREAD' in reason

    def test_exactly_2_5x_fails(self):
        """2.5x is the limit — must fail (< not <=)."""
        f = {'spread_bps': 5.0, 'spread_mean_20': 2.0}  # exactly 2.5x
        passed, _, _, _ = _gate_spread(f, GateThresholds())
        assert not passed

    def test_zero_mean_passes(self):
        """Zero mean should not cause divide-by-zero."""
        f = {'spread_bps': 1.0, 'spread_mean_20': 0.0}
        passed, _, _, _ = _gate_spread(f, GateThresholds())
        assert passed


class TestGate3BBSqueeze:
    """Gate 3: Bollinger Band squeeze."""

    def test_no_squeeze_passes(self):
        f = {'bb_squeeze': 0.0}
        passed, reason, _ = _gate_squeeze(f, GateThresholds())
        assert passed

    def test_squeeze_active_fails(self):
        f = {'bb_squeeze': 1.0}
        passed, reason, _ = _gate_squeeze(f, GateThresholds())
        assert not passed
        assert 'BB_SQUEEZE' in reason

    def test_missing_feature_passes(self):
        passed, _, _ = _gate_squeeze({}, GateThresholds())
        assert passed  # default: 0.0


class TestGate4Liquidation:
    """Gate 4: Liquidation cascade."""

    def test_normal_liq_ratio_passes(self):
        f = {'liq_ratio': 1.2, 'liq_total_usd': 100000}
        passed, _, _ = _gate_liquidation(f, GateThresholds())
        assert passed

    def test_extreme_long_cascade_fails(self):
        """liq_ratio > 5 = massive long liquidations."""
        f = {'liq_ratio': 6.0, 'liq_total_usd': 5000000}
        passed, reason, _ = _gate_liquidation(f, GateThresholds())
        assert not passed
        assert 'LIQUIDATION' in reason

    def test_extreme_short_cascade_fails(self):
        """liq_ratio < 0.2 = massive short liquidations."""
        f = {'liq_ratio': 0.1, 'liq_total_usd': 5000000}
        passed, reason, _ = _gate_liquidation(f, GateThresholds())
        assert not passed


class TestGate5OddsEV:
    """Gate 5: Expected Value from Polymarket odds."""

    def test_positive_ev_passes(self):
        """Model P(UP)=0.75, market implies 0.60 → EV=0.15 > 0.05."""
        passed, reason, ev = _gate_odds_ev(0.75, Signal.BET_UP_NORMAL, 1.667, GateThresholds())
        assert passed
        assert ev > 0.05

    def test_negative_ev_fails(self):
        """Model P(UP)=0.73, market implies 0.70 → EV=0.03 < 0.05."""
        passed, reason, ev = _gate_odds_ev(0.73, Signal.BET_UP_NORMAL, 1.43, GateThresholds())
        assert not passed
        assert 'ODDS_EV' in reason

    def test_ev_direction_down(self):
        """For DOWN bet: EV = (1 - model_prob) - implied_down."""
        # model says 0.15 prob UP → 0.85 prob DOWN
        # market implies DOWN at 0.70 → EV = 0.85 - 0.70 = 0.15
        passed, _, ev = _gate_odds_ev(0.15, Signal.BET_DOWN_HIGH_CONF, 1.429, GateThresholds())
        assert passed
        assert ev > 0.05

    def test_no_odds_passes_gate(self):
        """Fail-open: no odds available → pass gate."""
        passed, _, ev = _gate_odds_ev(0.75, Signal.BET_UP_NORMAL, 0.0, GateThresholds())
        assert passed  # pass by default

    def test_no_bet_signal_passes(self):
        """NO_BET signals don't need EV check."""
        passed, _, _ = _gate_odds_ev(0.50, Signal.NO_BET, 2.0, GateThresholds())
        assert passed


class TestGate6BotState:
    """Gate 6: Bot state."""

    def test_active_passes(self):
        passed, _ = _gate_bot_state(BotState.ACTIVE)
        assert passed

    def test_cooling_down_fails(self):
        passed, reason = _gate_bot_state(BotState.COOLING_DOWN)
        assert not passed
        assert 'BOT_STATE' in reason

    def test_paused_fails(self):
        passed, reason = _gate_bot_state(BotState.PAUSED)
        assert not passed

    def test_boot_fails(self):
        passed, _ = _gate_bot_state(BotState.BOOT)
        assert not passed


# ═══════════════════════════════════════════════════════════════════════════════
# RiskGates Orchestrator — All Gates Together
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskGatesOrchestrator:
    """All 6 gates evaluated together."""

    def _clean_features(self) -> dict:
        return {
            'vol_regime_zscore': 0.5,
            'atr_ratio': 1.0,
            'spread_bps': 1.5,
            'spread_mean_20': 2.0,
            'bb_squeeze': 0.0,
            'liq_ratio': 1.2,
            'liq_total_usd': 50000.0,
            'prob': 0.75,
        }

    def test_all_gates_pass(self):
        gates = RiskGates()
        result = gates.evaluate(
            self._clean_features(),
            signal=Signal.BET_UP_NORMAL,
            odds=1.667,
            bot_state=BotState.ACTIVE,
        )
        assert result.all_pass
        assert result.failures == []

    def test_all_gates_run_even_on_first_failure(self):
        """Gate 1 fails but gates 2-6 still evaluated."""
        gates = RiskGates()
        bad_features = {**self._clean_features(), 'vol_regime_zscore': 3.0}
        result = gates.evaluate(
            bad_features,
            signal=Signal.BET_UP_NORMAL,
            odds=1.667,
            bot_state=BotState.ACTIVE,
        )
        assert not result.all_pass
        assert not result.volatility_gate
        # Other gates still ran
        assert result.spread_gate is True
        assert result.squeeze_gate is True

    def test_multiple_failures_all_reported(self):
        """All failing gates reported, not just first."""
        gates = RiskGates()
        bad_features = {
            **self._clean_features(),
            'vol_regime_zscore': 3.0,   # Gate 1 fail
            'spread_bps': 10.0,          # Gate 2 fail
            'bb_squeeze': 1.0,           # Gate 3 fail
        }
        result = gates.evaluate(bad_features)
        assert len(result.failures) == 3

    def test_cooling_down_fails_gate_6(self):
        gates = RiskGates()
        result = gates.evaluate(
            self._clean_features(),
            bot_state=BotState.COOLING_DOWN,
        )
        assert not result.all_pass
        assert not result.bot_state_gate
        assert any('BOT_STATE' in f for f in result.failures)

    def test_gate_result_to_dict(self):
        gates = RiskGates()
        result = gates.evaluate(self._clean_features(), bot_state=BotState.ACTIVE)
        d = result.to_dict()
        assert 'all_pass' in d
        assert 'gate_volatility' in d
        assert 'diagnostics' in d

    def test_stats_tracking(self):
        gates = RiskGates()
        # 2 evaluations, 1 fails
        gates.evaluate(self._clean_features(), bot_state=BotState.ACTIVE)
        gates.evaluate(self._clean_features(), bot_state=BotState.COOLING_DOWN)
        stats = gates.stats()
        assert stats['total_evaluations'] == 2
        assert stats['failure_counts']['bot_state'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Odds Checker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOddsChecker:
    """Test EV computation and fail-open behaviour."""

    def test_ev_computation_up(self):
        checker = OddsChecker(ev_minimum=0.05)
        checker._latest_odds = OddsData(
            yes_price=0.60, no_price=0.40,
            odds_up=1.667, odds_down=2.5,
            implied_prob_up=0.60, implied_prob_down=0.40,
            fetched_at=__import__('time').time(),
            source='live',
        )
        ev, passes = checker.compute_ev(0.75, 'UP')
        assert abs(ev - 0.15) < 0.001   # 0.75 - 0.60 = 0.15
        assert passes

    def test_ev_computation_down(self):
        checker = OddsChecker(ev_minimum=0.05)
        checker._latest_odds = OddsData(
            yes_price=0.55, no_price=0.45,
            implied_prob_up=0.55, implied_prob_down=0.45,
            fetched_at=__import__('time').time(), source='live',
        )
        # model says 0.20 UP → 0.80 DOWN. market DOWN at 0.45 → EV = 0.80 - 0.45 = 0.35
        ev, passes = checker.compute_ev(0.20, 'DOWN')
        assert ev > 0.05
        assert passes

    def test_no_odds_passes(self):
        """No cached odds → fail-open."""
        checker = OddsChecker()
        ev, passes = checker.compute_ev(0.75, 'UP')
        assert passes  # fail-open

    def test_negative_ev_fails(self):
        checker = OddsChecker(ev_minimum=0.05)
        checker._latest_odds = OddsData(
            yes_price=0.72, no_price=0.28,
            implied_prob_up=0.72, implied_prob_down=0.28,
            fetched_at=__import__('time').time(), source='live',
        )
        # model says 0.73, market implies 0.72 → EV = 0.01 < 0.05
        ev, passes = checker.compute_ev(0.73, 'UP')
        assert not passes
        assert abs(ev - 0.01) < 0.001

    def test_odds_data_freshness(self):
        import time
        odds = OddsData(fetched_at=time.time(), yes_price=0.5, no_price=0.5)
        assert odds.is_fresh
        odds_stale = OddsData(fetched_at=time.time() - 400, yes_price=0.5, no_price=0.5)
        assert not odds_stale.is_fresh

    def test_stats(self):
        checker = OddsChecker()
        s = checker.stats()
        assert 'fetch_count' in s
        assert 'success_rate' in s


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Enums Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalEnum:
    def test_is_bet(self):
        assert Signal.BET_UP_NORMAL.is_bet
        assert Signal.BET_UP_HIGH_CONF.is_bet
        assert not Signal.NO_BET.is_bet

    def test_is_high_conf(self):
        assert Signal.BET_UP_HIGH_CONF.is_high_conf
        assert Signal.BET_DOWN_HIGH_CONF.is_high_conf
        assert not Signal.BET_UP_NORMAL.is_high_conf

    def test_direction(self):
        assert Signal.BET_UP_NORMAL.direction == 'UP'
        assert Signal.BET_DOWN_NORMAL.direction == 'DOWN'
        assert Signal.NO_BET.direction == 'NONE'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
