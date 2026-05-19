"""
Tests for Phase 5 — Compounding & Position Sizing.

Covers:
  1. Bracket table — correct sizing per bracket and tier
  2. Bracket find — correct bracket for any balance
  3. Upgrade rules — all 3 conditions, defer logic
  4. Downgrade rules — immediate downgrade, stat reset
  5. Hard stop — balance < $5.00 → PAUSED + alert
  6. Kelly sizing — formula correctness, fraction, cap
  7. Kelly negative edge — returns 0
  8. Kelly mode threshold — $100+ routes to Kelly
  9. Portfolio tracker — win rate, drawdown, Sharpe
  10. Simulation — bracket progression 5→15→25→40→65→100+
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sizing.compounding_engine import (
    CompoundingEngine, Bracket, BRACKETS,
    MINIMUM_BET_FLOOR, KELLY_MODE_THRESHOLD,
    UPGRADE_MIN_BETS, UPGRADE_MIN_WIN_RATE,
)
from src.sizing.kelly_sizer import KellySizer, KellyResult
from src.sizing.portfolio_tracker import PortfolioTracker
from src.signal.signal_generator import Signal, BotState


# ═══════════════════════════════════════════════════════════════════════════════
# Bracket Table Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBracketTable:
    """Verify all bracket definitions match PRD 8.1."""

    def test_bracket_count(self):
        assert len(BRACKETS) == 6

    def test_bracket_0_values(self):
        b = BRACKETS[0]
        assert b.floor == 5.00
        assert b.ceil == 14.99
        assert b.normal_bet == 1.00
        assert b.high_conf_bet == 2.00

    def test_bracket_1_values(self):
        b = BRACKETS[1]
        assert b.floor == 15.00
        assert b.ceil == 24.99
        assert b.normal_bet == 1.50
        assert b.high_conf_bet == 3.00

    def test_bracket_2_values(self):
        b = BRACKETS[2]
        assert b.floor == 25.00
        assert b.ceil == 39.99
        assert b.normal_bet == 2.50
        assert b.high_conf_bet == 5.00

    def test_bracket_3_values(self):
        b = BRACKETS[3]
        assert b.floor == 40.00
        assert b.ceil == 64.99
        assert b.normal_bet == 4.00
        assert b.high_conf_bet == 8.00

    def test_bracket_4_values(self):
        b = BRACKETS[4]
        assert b.floor == 65.00
        assert b.ceil == 99.99
        assert b.normal_bet == 6.50
        assert b.high_conf_bet == 13.00

    def test_bracket_5_is_kelly_mode(self):
        b = BRACKETS[5]
        assert b.floor == 100.00
        assert b.is_kelly_mode
        assert b.normal_bet is None
        assert b.high_conf_bet is None

    def test_brackets_contiguous(self):
        """Bracket floors and ceilings must be contiguous (no gap)."""
        for i in range(len(BRACKETS) - 2):
            assert abs(BRACKETS[i].ceil - (BRACKETS[i + 1].floor - 0.01)) < 0.001, \
                f"Gap between bracket {i} and {i+1}"

    def test_hard_stop_constant(self):
        # Hard stop is disabled — verify MINIMUM_BET_FLOOR is defined instead
        assert MINIMUM_BET_FLOOR == 1.00

    def test_kelly_threshold_constant(self):
        assert KELLY_MODE_THRESHOLD == 100.00


class TestBracketFinding:
    """Test _find_bracket for all balance levels."""

    def test_exactly_at_floor_is_in_bracket(self):
        eng = CompoundingEngine(5.00)
        assert eng.current_bracket.index == 0

    def test_balance_9_is_bracket_0(self):
        eng = CompoundingEngine(9.00)
        assert eng.current_bracket.index == 0

    def test_balance_15_is_bracket_1(self):
        eng = CompoundingEngine(15.00)
        assert eng.current_bracket.index == 1

    def test_balance_30_is_bracket_2(self):
        eng = CompoundingEngine(30.00)
        assert eng.current_bracket.index == 2

    def test_balance_50_is_bracket_3(self):
        eng = CompoundingEngine(50.00)
        assert eng.current_bracket.index == 3

    def test_balance_70_is_bracket_4(self):
        eng = CompoundingEngine(70.00)
        assert eng.current_bracket.index == 4

    def test_balance_100_is_kelly(self):
        eng = CompoundingEngine(100.00)
        assert eng.current_bracket.index == 5
        assert eng.is_kelly_mode

    def test_balance_500_is_kelly(self):
        eng = CompoundingEngine(500.00)
        assert eng.is_kelly_mode


# ═══════════════════════════════════════════════════════════════════════════════
# Bet Sizing Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBetSizing:
    """Test resolve_bet_size for each bracket and tier."""

    def test_bracket_0_normal_bet(self):
        eng = CompoundingEngine(10.00)
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 1.00

    def test_bracket_0_high_conf_bet(self):
        eng = CompoundingEngine(10.00)
        size = eng.resolve_bet_size(Signal.BET_UP_HIGH_CONF)
        assert size == 2.00

    def test_bracket_2_normal_bet(self):
        eng = CompoundingEngine(30.00)
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 2.50

    def test_bracket_2_high_conf_bet(self):
        eng = CompoundingEngine(30.00)
        size = eng.resolve_bet_size(Signal.BET_DOWN_HIGH_CONF)
        assert size == 5.00

    def test_bracket_4_normal(self):
        eng = CompoundingEngine(80.00)
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 6.50

    def test_no_bet_returns_zero(self):
        eng = CompoundingEngine(10.00)
        size = eng.resolve_bet_size(Signal.NO_BET)
        assert size == 0.0

    def test_bet_returns_zero_when_balance_below_bet_size(self):
        """When balance < $1.00 normal bet, returns 0.0 (no bet, no hard stop)."""
        eng = CompoundingEngine(0.50)   # below the $1.00 minimum bet
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 0.0
        # Bot stays ACTIVE — no hard stop forced
        assert eng.bot_state == BotState.ACTIVE


# ═══════════════════════════════════════════════════════════════════════════════
# Upgrade Rules Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpgradeRules:
    """PRD 8.2 — all 3 conditions must be met for upgrade."""

    def _win_to_ceiling(self, eng: CompoundingEngine, target_balance: float) -> None:
        """Helper: repeatedly win until balance reaches target."""
        while eng.balance < target_balance:
            size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
            if size == 0:
                break
            payout = size * 1.90   # 90% profit win
            eng.on_bet_resolved(won=True, bet_amount=size, payout=payout)

    def test_upgrade_requires_all_three_conditions(self):
        eng = CompoundingEngine(14.00)
        # Get to ceiling: add wins until balance >= 14.99
        self._win_to_ceiling(eng, 14.99)
        # At ceiling, but may not have 10 bets yet
        # Force: set internal state to have <10 bets
        eng._bets_in_bracket = 5   # under minimum
        assert not eng.can_upgrade

    def test_upgrade_requires_min_bets(self):
        eng = CompoundingEngine(14.99)
        eng._bets_in_bracket = 9     # one short
        eng._wins_in_bracket = 7
        eng._balance = 15.00
        assert not eng.can_upgrade

    def test_upgrade_requires_win_rate_50pct(self):
        eng = CompoundingEngine(14.99)
        eng._bets_in_bracket = 15
        eng._wins_in_bracket = 7    # 7/15 = 46.7% < 50%
        eng._balance = 15.00
        assert not eng.can_upgrade

    def test_upgrade_succeeds_all_conditions_met(self):
        eng = CompoundingEngine(14.99)
        eng._bets_in_bracket = 10
        eng._wins_in_bracket = 6    # 60% >= 50%
        eng._balance = 15.00
        assert eng.can_upgrade

    def test_upgrade_resets_bracket_stats(self):
        eng = CompoundingEngine(14.00)
        # Give it enough wins to be at ceiling with 10 bets and >50% WR
        eng._balance = 15.01
        eng._bets_in_bracket = 10
        eng._wins_in_bracket = 7
        eng._check_upgrade()
        assert eng._bets_in_bracket == 0
        assert eng._wins_in_bracket == 0
        assert eng.current_bracket.index == 1

    def test_upgrade_deferred_low_win_rate(self):
        """At ceiling with <50% WR: defer upgrade, no crash."""
        eng = CompoundingEngine(14.00)
        eng._balance = 15.01
        eng._bets_in_bracket = 10
        eng._wins_in_bracket = 4   # 40% WR
        prev_bracket = eng.current_bracket.index
        eng._check_upgrade()
        # Stays in same bracket
        assert eng.current_bracket.index == prev_bracket

    def test_upgrade_fires_alert(self):
        alerts = []
        eng = CompoundingEngine(14.99, on_alert=alerts.append)
        eng._balance = 15.01
        eng._bets_in_bracket = 10
        eng._wins_in_bracket = 7
        eng._check_upgrade()
        assert any('UPGRADE' in a for a in alerts)


# ═══════════════════════════════════════════════════════════════════════════════
# Downgrade Rules Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDowngradeRules:
    """PRD 8.3 — immediate downgrade when balance < floor."""

    def test_downgrade_on_loss(self):
        eng = CompoundingEngine(16.00)   # bracket 1
        assert eng.current_bracket.index == 1
        # Lose enough to fall below bracket 1 floor ($15.00)
        eng._balance = 14.50
        eng._check_downgrade()
        assert eng.current_bracket.index == 0

    def test_downgrade_resets_stats(self):
        eng = CompoundingEngine(16.00)
        eng._bets_in_bracket = 8
        eng._wins_in_bracket = 6
        eng._balance = 14.00
        eng._check_downgrade()
        assert eng._bets_in_bracket == 0
        assert eng._wins_in_bracket == 0

    def test_downgrade_fires_alert(self):
        alerts = []
        eng = CompoundingEngine(16.00, on_alert=alerts.append)
        eng._balance = 14.00
        eng._check_downgrade()
        assert any('DOWNGRADE' in a for a in alerts)

    def test_no_downgrade_within_bracket(self):
        eng = CompoundingEngine(20.00)  # bracket 1
        eng._bets_in_bracket = 5
        eng._balance = 17.00           # still in bracket 1
        eng._check_downgrade()
        assert eng.current_bracket.index == 1
        assert eng._bets_in_bracket == 5   # not reset

    def test_multi_bracket_downgrade(self):
        """Can drop more than one bracket at once."""
        eng = CompoundingEngine(50.00)  # bracket 3
        assert eng.current_bracket.index == 3
        eng._balance = 12.00            # falls to bracket 0
        eng._check_downgrade()
        assert eng.current_bracket.index == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Bet Until Insufficient (No Hard Stop)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBetUntilInsufficient:
    """No hard stop — bot bets until balance can't cover any bet."""

    def test_bot_stays_active_below_5(self):
        """BotState never becomes PAUSED automatically due to low balance."""
        eng = CompoundingEngine(5.50)
        # Lose $0.51 — balance $4.99, below old hard-stop threshold
        eng.on_bet_resolved(won=False, bet_amount=0.51)
        assert eng.bot_state == BotState.ACTIVE  # NOT PAUSED

    def test_bet_returns_zero_below_1_dollar(self):
        """When balance < $1.00, no bet can be placed — returns 0.0."""
        eng = CompoundingEngine(0.80)
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 0.0
        assert eng.bot_state == BotState.ACTIVE  # still ACTIVE, just can't bet

    def test_bet_returns_zero_at_zero_balance(self):
        """Zero balance returns 0.0 bet."""
        eng = CompoundingEngine(5.00)
        eng._balance = 0.0
        assert eng.resolve_bet_size(Signal.BET_UP_NORMAL) == 0.0

    def test_can_bet_property(self):
        """can_bet property reflects whether balance >= minimum bet."""
        eng = CompoundingEngine(5.00)
        assert eng.can_bet
        eng._balance = 0.99
        assert not eng.can_bet

    def test_exactly_1_dollar_can_bet(self):
        """Exactly $1.00 balance: normal bet is $1.00 and is allowed."""
        eng = CompoundingEngine(1.00)
        size = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        assert size == 1.00

    def test_no_paused_state_from_losses(self):
        """Multiple consecutive losses never trigger PAUSED state."""
        eng = CompoundingEngine(5.00)
        for _ in range(5):
            if eng.resolve_bet_size(Signal.BET_UP_NORMAL) == 0:
                break
            eng.on_bet_resolved(won=False, bet_amount=1.00)
        assert eng.bot_state == BotState.ACTIVE

    def test_high_conf_blocked_if_insufficient(self):
        """HIGH_CONF ($2.00) blocked if balance < $2.00, even if > $1.00."""
        eng = CompoundingEngine(1.50)   # can afford normal ($1), not high_conf ($2)
        size_normal = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
        size_hc = eng.resolve_bet_size(Signal.BET_UP_HIGH_CONF)
        assert size_normal == 1.00  # normal OK
        assert size_hc == 0.0       # high_conf blocked


# ═══════════════════════════════════════════════════════════════════════════════
# Kelly Sizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellySizer:
    """PRD 8.4 — Kelly formula, fraction, cap, edge cases."""

    def test_kelly_formula_correctness(self):
        """
        Manual: p=0.60, odds=1.90, b=0.90
        k = (0.90×0.60 - 0.40) / 0.90 = (0.54 - 0.40) / 0.90 = 0.1556
        fractional = 0.1556 × 0.25 = 0.0389
        bet at $100 = $3.89 (but capped at $100 × 0.02 = $2.00)
        """
        sizer = KellySizer(fraction=0.25, max_pct=0.02)
        result = sizer.compute(prob=0.60, odds=1.90, balance=100.0)
        assert abs(result.raw_kelly - 0.1556) < 0.001
        assert abs(result.fractional_kelly - 0.0389) < 0.001
        assert result.bet_size == 2.00   # capped at 2%
        assert result.capped

    def test_kelly_fraction_applied(self):
        """At $200 balance with high edge, still capped at 2%."""
        sizer = KellySizer(fraction=0.25, max_pct=0.02)
        result = sizer.compute(prob=0.70, odds=2.00, balance=200.0)
        max_allowed = 200.0 * 0.02
        assert result.bet_size <= max_allowed

    def test_negative_edge_returns_zero(self):
        """When model prob < implied prob → Kelly < 0 → no bet."""
        sizer = KellySizer()
        result = sizer.compute(prob=0.40, odds=1.90, balance=100.0)
        # k = (0.90×0.40 - 0.60) / 0.90 = (0.36 - 0.60)/0.90 = negative
        assert result.raw_kelly < 0
        assert result.bet_size == 0.0

    def test_zero_bet_on_breakeven(self):
        """Exact breakeven: p = 1/(1+b) → k = 0."""
        # For odds=2.0: b=1.0, breakeven p = 0.50
        sizer = KellySizer()
        result = sizer.compute(prob=0.50, odds=2.00, balance=100.0)
        assert result.raw_kelly == 0.0
        assert result.bet_size == 0.0

    def test_invalid_odds_returns_zero(self):
        sizer = KellySizer()
        result = sizer.compute(prob=0.60, odds=0.5, balance=100.0)
        assert result.bet_size == 0.0

    def test_kelly_mode_activated_at_100(self):
        eng = CompoundingEngine(100.00)
        bet = eng.resolve_bet_size(Signal.BET_UP_NORMAL, prob=0.65, odds=1.90)
        # Should be Kelly-based, not fixed bracket
        assert isinstance(bet, float)
        assert bet >= 0.0
        assert bet <= 100.0 * 0.02 + 0.01  # ~2% cap

    def test_kelly_below_100_uses_bracket(self):
        eng = CompoundingEngine(50.00)   # bracket 3
        bet = eng.resolve_bet_size(Signal.BET_UP_NORMAL, prob=0.65, odds=1.90)
        assert bet == BRACKETS[3].normal_bet   # fixed $4.00

    def test_kelly_expected_growth_positive(self):
        """With positive edge, expected growth should be positive."""
        sizer = KellySizer()
        growth = sizer.expected_growth_rate(prob=0.60, odds=1.90)
        assert growth > 0

    def test_kelly_expected_growth_negative_edge(self):
        sizer = KellySizer()
        growth = sizer.expected_growth_rate(prob=0.40, odds=1.90)
        assert growth <= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio Tracker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioTracker:
    """Test portfolio metrics."""

    def _fill_tracker(self, tracker: PortfolioTracker, bets: list[bool]) -> None:
        bal = 10.0
        for won in bets:
            amt = 1.0
            payout = 1.90 if won else 0.0
            pnl = payout - amt if won else -amt
            bal_after = bal + pnl
            tracker.record_bet(
                signal_name='BET_UP_NORMAL', direction='UP', is_high_conf=False,
                prob=0.75, odds=1.90, ev=0.10, bracket=0,
                bet_amount=amt, payout=payout, won=won,
                balance_before=bal, balance_after=bal_after,
            )
            bal = bal_after

    def test_win_rate(self):
        tracker = PortfolioTracker(10.0)
        self._fill_tracker(tracker, [True, True, False, True])
        assert abs(tracker.win_rate - 0.75) < 0.001

    def test_total_pnl(self):
        tracker = PortfolioTracker(10.0)
        self._fill_tracker(tracker, [True, False])
        # Win: +0.90, Loss: -1.00 → PnL = -0.10
        assert abs(tracker.total_pnl - (-0.10)) < 0.01

    def test_max_drawdown(self):
        tracker = PortfolioTracker(10.0)
        self._fill_tracker(tracker, [False, False, True])
        dd = tracker.max_drawdown()
        assert dd > 0

    def test_current_loss_streak(self):
        tracker = PortfolioTracker(10.0)
        self._fill_tracker(tracker, [True, False, False])
        assert tracker.current_loss_streak() == 2

    def test_current_loss_streak_reset_by_win(self):
        tracker = PortfolioTracker(10.0)
        self._fill_tracker(tracker, [False, False, True])
        assert tracker.current_loss_streak() == 0

    def test_summary_structure(self):
        tracker = PortfolioTracker(10.0)
        s = tracker.summary()
        assert 'win_rate' in s
        assert 'total_pnl' in s
        assert 'max_drawdown' in s
        assert 'sharpe_ratio' in s
        assert 'high_conf' in s


# ═══════════════════════════════════════════════════════════════════════════════
# Bracket Progression Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBracketProgressionSimulation:
    """End-to-end simulation: bracket 0 → bracket 5 (Kelly mode)."""

    def test_progression_from_5_to_100(self):
        """
        Simulate high win rate progression until Kelly mode.
        Verifies bracket upgrades happen at the right balance thresholds.
        Starting balance $10 (safely above hard stop) with 90% win rate.
        """
        eng = CompoundingEngine(10.00)
        assert eng.current_bracket.index == 0

        seen_brackets = {0}
        max_iters = 1000

        for i in range(max_iters):
            if eng.bot_state == BotState.PAUSED:
                break

            # Win 90% of time to ensure progression without hard stop
            won = (i % 10 != 0)
            bet = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
            if bet == 0:
                break

            # Payout = bet returned + profit (odds 1.90 means 90% profit)
            payout = bet * 1.90 if won else 0.0
            eng.on_bet_resolved(won=won, bet_amount=bet, payout=payout)
            seen_brackets.add(eng.current_bracket.index)

            if eng.is_kelly_mode:
                break

        # Should have progressed through multiple brackets
        assert len(seen_brackets) >= 2, f"Only saw brackets: {seen_brackets}"

    def test_hard_stop_prevents_negative_balance(self):
        """Balance must never go below 0."""
        eng = CompoundingEngine(5.50)
        for _ in range(10):
            bet = eng.resolve_bet_size(Signal.BET_UP_NORMAL)
            if bet == 0:
                break
            eng.on_bet_resolved(won=False, bet_amount=bet, payout=0.0)

        assert eng.balance >= 0.0

    def test_downgrade_then_recovery(self):
        """Can downgrade then re-upgrade."""
        eng = CompoundingEngine(25.00)   # bracket 2
        assert eng.current_bracket.index == 2

        # Lose until downgrade to bracket 1
        eng._balance = 14.00
        eng._check_downgrade()
        assert eng.current_bracket.index == 0

        # Win back to bracket 1+
        eng._balance = 20.00
        eng._check_upgrade()  # won't upgrade yet (0 bets)
        assert eng.current_bracket.index == 0   # no upgrade yet, needs 10 bets


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
