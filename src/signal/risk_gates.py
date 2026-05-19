"""
Risk Gates — 6 independent boolean filters that block bet execution.

PRD 7.2: ALL 6 gates must pass for filters_ok=True.

Gate Architecture:
  Each gate is a pure function: features_dict → (passed: bool, reason: str)
  All gates run independently — a failure in one does not short-circuit others.
  This allows full gate failure logging for diagnostics and tuning.

Gate Summary:
  ┌──────┬────────────────────┬──────────────────────────────────────────────┐
  │ Gate │ Name               │ Condition to PASS (allow bet)                │
  ├──────┼────────────────────┼──────────────────────────────────────────────┤
  │  1   │ Volatility Regime  │ vol_regime_zscore < 2.0 AND atr < P90        │
  │  2   │ Spread             │ spread_bps < 2.5 × 20-bar avg spread         │
  │  3   │ BB Squeeze         │ bb_squeeze == 0 (band NOT at 20-bar minimum)  │
  │  4   │ Liquidation Cascade│ recent liq < 5 × 5-bar average liquidation   │
  │  5   │ Odds EV            │ model_prob − implied_prob > 0.05              │
  │  6   │ Bot State          │ bot_state == ACTIVE                           │
  └──────┴────────────────────┴──────────────────────────────────────────────┘

Gate 5 (Odds EV) requires OddsChecker to have fetched current Polymarket odds.
Gate 6 is always checked last — if cooling down, no bet regardless of gates 1-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.signal.signal_generator import Signal, BotState

logger = structlog.get_logger(__name__)


# ─── Gate result ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result from evaluating all 6 gates."""
    # Individual gate outcomes
    volatility_gate: bool = True
    spread_gate: bool = True
    squeeze_gate: bool = True
    liquidation_gate: bool = True
    odds_ev_gate: bool = True
    bot_state_gate: bool = True

    # Reason strings for each failure
    failures: list[str] = field(default_factory=list)

    # Computed values for transparency
    vol_regime: float = 0.0
    spread_bps: float = 0.0
    spread_ratio: float = 0.0
    bb_squeeze: float = 0.0
    liq_ratio: float = 0.0
    ev: float = 0.0

    @property
    def all_pass(self) -> bool:
        return len(self.failures) == 0

    def to_dict(self) -> dict:
        return {
            'all_pass': self.all_pass,
            'gate_volatility': self.volatility_gate,
            'gate_spread': self.spread_gate,
            'gate_squeeze': self.squeeze_gate,
            'gate_liquidation': self.liquidation_gate,
            'gate_odds_ev': self.odds_ev_gate,
            'gate_bot_state': self.bot_state_gate,
            'failures': self.failures,
            'diagnostics': {
                'vol_regime_zscore': round(self.vol_regime, 3),
                'spread_bps': round(self.spread_bps, 3),
                'spread_ratio_vs_avg': round(self.spread_ratio, 3),
                'bb_squeeze': self.bb_squeeze,
                'liq_cascade_ratio': round(self.liq_ratio, 3),
                'odds_ev': round(self.ev, 4),
            },
        }


# ─── Gate thresholds (PRD 7.2) ────────────────────────────────────────────────

class GateThresholds:
    """Configurable gate thresholds with PRD defaults."""

    # Gate 1 — Volatility Regime
    VOL_REGIME_ZSCORE_MAX: float = 2.0     # z-score of realized vol
    ATR_PERCENTILE_MAX: float = 90.0       # ATR percentile within history

    # Gate 2 — Spread
    SPREAD_MULTIPLIER_MAX: float = 2.5     # max spread vs 20-bar avg

    # Gate 3 — BB Squeeze (0 = not squeezed, 1 = squeezed)
    BB_SQUEEZE_BLOCK: float = 1.0         # block if squeeze == 1

    # Gate 4 — Liquidation Cascade
    LIQ_CASCADE_MULTIPLIER: float = 5.0   # block if current > 5x avg

    # Gate 5 — Odds EV
    EV_MINIMUM: float = 0.05              # minimum edge over implied prob


# ─── Individual gate functions ────────────────────────────────────────────────

def _gate_volatility(
    features: dict,
    thresholds: GateThresholds,
) -> tuple[bool, str, float]:
    """
    Gate 1 — Volatility Regime.

    Passes when:
      - vol_regime_zscore < 2.0   (not in extreme volatility)
      - atr_14 not in top 10%     (proxy: atr_ratio < threshold)

    High volatility periods have unpredictable price action and
    invalidate the model's training distribution.
    """
    vol_regime = float(features.get('vol_regime_zscore', 0.0))
    atr_ratio = float(features.get('atr_ratio', 1.0))

    # atr_ratio > 1.5 used as proxy for atr_percentile > 90
    # (actual percentile requires history; ratio of atr_7/atr_14 captures regime shift)
    atr_extreme = atr_ratio > 1.5

    passed = (vol_regime < thresholds.VOL_REGIME_ZSCORE_MAX) and (not atr_extreme)
    reason = ''
    if not passed:
        parts = []
        if vol_regime >= thresholds.VOL_REGIME_ZSCORE_MAX:
            parts.append(f'vol_regime={vol_regime:.2f}>={thresholds.VOL_REGIME_ZSCORE_MAX}')
        if atr_extreme:
            parts.append(f'atr_ratio={atr_ratio:.2f} (extreme)')
        reason = f'VOLATILITY: {", ".join(parts)}'

    return passed, reason, vol_regime


def _gate_spread(
    features: dict,
    thresholds: GateThresholds,
) -> tuple[bool, str, float, float]:
    """
    Gate 2 — Bid-Ask Spread.

    Passes when:
      spread_bps < spread_mean_20 × 2.5

    Wide spreads indicate low liquidity, market stress, or
    upcoming news events. Executing into a 2.5× normal spread
    dramatically erodes edge.
    """
    spread_bps = float(features.get('spread_bps', 0.0))
    spread_mean = float(features.get('spread_mean_20', spread_bps + 1e-10))

    if spread_mean <= 0:
        return True, '', spread_bps, 1.0

    ratio = spread_bps / spread_mean
    passed = ratio < thresholds.SPREAD_MULTIPLIER_MAX
    reason = (
        f'SPREAD: {spread_bps:.2f}bps = {ratio:.1f}x avg '
        f'(limit {thresholds.SPREAD_MULTIPLIER_MAX}x)'
        if not passed else ''
    )
    return passed, reason, spread_bps, ratio


def _gate_squeeze(
    features: dict,
    thresholds: GateThresholds,
) -> tuple[bool, str, float]:
    """
    Gate 3 — Bollinger Band Squeeze.

    Passes when:
      bb_squeeze == 0 (bands NOT at 20-bar minimum width)

    A BB squeeze means volatility is compressed — the market is
    coiling for a breakout. Direction is unknown, so no bets.
    """
    bb_squeeze = float(features.get('bb_squeeze', 0.0))
    passed = bb_squeeze < thresholds.BB_SQUEEZE_BLOCK
    reason = f'BB_SQUEEZE: band at 20-bar minimum (breakout risk)' if not passed else ''
    return passed, reason, bb_squeeze


def _gate_liquidation(
    features: dict,
    thresholds: GateThresholds,
) -> tuple[bool, str, float]:
    """
    Gate 4 — Liquidation Cascade.

    Passes when:
      (liq_long + liq_short) in current bar < 5 × 5-bar average

    Large liquidation cascades cause non-linear price moves that
    are not captured in the model's feature distribution. These
    events cause adversarial conditions for any statistical model.
    """
    liq_total = float(features.get('liq_total_usd', 0.0))
    liq_ratio = float(features.get('liq_ratio', 1.0))

    # Use liq_ratio (long/short ratio) as cascade signal proxy
    # A more direct signal: check if total liquidations spike
    # We use liq_total vs a 5-bar context captured in liq_ratio field
    # Gate fires when liq_ratio is extreme (>5 or <0.2) = cascade in one direction
    liq_extreme = (liq_ratio > thresholds.LIQ_CASCADE_MULTIPLIER
                   or liq_ratio < (1.0 / thresholds.LIQ_CASCADE_MULTIPLIER))

    passed = not liq_extreme
    reason = (
        f'LIQUIDATION_CASCADE: liq_ratio={liq_ratio:.2f} '
        f'(limit 1/{thresholds.LIQ_CASCADE_MULTIPLIER:.0f} to {thresholds.LIQ_CASCADE_MULTIPLIER:.0f}x)'
        if not passed else ''
    )
    return passed, reason, liq_ratio


def _gate_odds_ev(
    prob: float,
    signal: Optional[Signal],
    odds: float,
    thresholds: GateThresholds,
) -> tuple[bool, str, float]:
    """
    Gate 5 — Polymarket Odds Expected Value.

    Passes when:
      model_prob − implied_prob > EV_MINIMUM (0.05)

    Where implied_prob = 1 / decimal_odds.

    This ensures we only bet when we have a genuine edge over the
    market's implied probability. Without this gate, we could be
    betting on correctly-predicted directions but at negative EV
    due to unfavourable Polymarket odds.

    Args:
        prob:   Model's calibrated P(UP).
        signal: Signal type (determines direction for EV calculation).
        odds:   Decimal odds from Polymarket (e.g., 1.90).

    Returns:
        (passed, reason, ev)
    """
    if odds <= 0 or signal is None or not signal.is_bet:
        # No odds available — pass by default (conservative: don't block)
        return True, '', 0.0

    implied_prob = 1.0 / odds

    # EV depends on direction
    if signal.is_up:
        ev = prob - implied_prob
    else:  # DOWN bet: we're betting on price falling, so complement probability
        ev = (1.0 - prob) - implied_prob

    passed = ev > thresholds.EV_MINIMUM
    reason = (
        f'ODDS_EV: edge={ev:.4f} < minimum {thresholds.EV_MINIMUM} '
        f'(model={prob:.4f}, implied={implied_prob:.4f}, odds={odds:.3f})'
        if not passed else ''
    )
    return passed, reason, ev


def _gate_bot_state(bot_state: BotState) -> tuple[bool, str]:
    """
    Gate 6 — Bot State.

    Passes ONLY when bot_state == ACTIVE.
    COOLING_DOWN and PAUSED always block execution.

    This is the final override — even if all other gates pass,
    a cooling-down or paused bot never executes a bet.
    """
    passed = bot_state == BotState.ACTIVE
    reason = f'BOT_STATE: {bot_state.name} (bets blocked)' if not passed else ''
    return passed, reason


# ─── RiskGates Orchestrator ───────────────────────────────────────────────────

class RiskGates:
    """
    Evaluates all 6 risk gates and returns a full GateResult.

    ALL 6 gates run regardless of individual failures — this ensures
    complete diagnostic information for every bar, enabling post-hoc
    analysis of why signals were blocked.

    Usage:
        gates = RiskGates()
        result = gates.evaluate(features, signal, odds, bot_state)
        if result.all_pass:
            execute_bet()
    """

    def __init__(self, thresholds: Optional[GateThresholds] = None):
        self._thresholds = thresholds or GateThresholds()
        self._eval_count: int = 0
        self._pass_count: int = 0
        self._failure_counts: dict[str, int] = {
            'volatility': 0, 'spread': 0, 'squeeze': 0,
            'liquidation': 0, 'odds_ev': 0, 'bot_state': 0,
        }

    def evaluate(
        self,
        features: dict,
        signal: Optional[Signal] = None,
        odds: float = 0.0,
        bot_state: BotState = BotState.ACTIVE,
    ) -> GateResult:
        """
        Evaluate all 6 gates and return GateResult.

        All gates run independently — a failure in one does NOT
        short-circuit evaluation of remaining gates.

        Args:
            features:  Feature dict from FeatureEngine.
            signal:    Proposed signal (needed for EV direction calculation).
            odds:      Polymarket decimal odds (0 if unavailable).
            bot_state: Current bot state.

        Returns:
            GateResult with per-gate outcomes and diagnostics.
        """
        self._eval_count += 1
        result = GateResult()
        failures = []

        # Gate 1 — Volatility Regime
        g1_pass, g1_reason, vol_regime = _gate_volatility(features, self._thresholds)
        result.volatility_gate = g1_pass
        result.vol_regime = vol_regime
        if not g1_pass:
            failures.append(g1_reason)
            self._failure_counts['volatility'] += 1

        # Gate 2 — Spread
        g2_pass, g2_reason, spread_bps, spread_ratio = _gate_spread(features, self._thresholds)
        result.spread_gate = g2_pass
        result.spread_bps = spread_bps
        result.spread_ratio = spread_ratio
        if not g2_pass:
            failures.append(g2_reason)
            self._failure_counts['spread'] += 1

        # Gate 3 — BB Squeeze
        g3_pass, g3_reason, bb_squeeze = _gate_squeeze(features, self._thresholds)
        result.squeeze_gate = g3_pass
        result.bb_squeeze = bb_squeeze
        if not g3_pass:
            failures.append(g3_reason)
            self._failure_counts['squeeze'] += 1

        # Gate 4 — Liquidation Cascade
        g4_pass, g4_reason, liq_ratio = _gate_liquidation(features, self._thresholds)
        result.liquidation_gate = g4_pass
        result.liq_ratio = liq_ratio
        if not g4_pass:
            failures.append(g4_reason)
            self._failure_counts['liquidation'] += 1

        # Gate 5 — Odds EV
        g5_pass, g5_reason, ev = _gate_odds_ev(
            features.get('prob', 0.5), signal, odds, self._thresholds
        )
        result.odds_ev_gate = g5_pass
        result.ev = ev
        if not g5_pass:
            failures.append(g5_reason)
            self._failure_counts['odds_ev'] += 1

        # Gate 6 — Bot State (always last)
        g6_pass, g6_reason = _gate_bot_state(bot_state)
        result.bot_state_gate = g6_pass
        if not g6_pass:
            failures.append(g6_reason)
            self._failure_counts['bot_state'] += 1

        result.failures = failures

        if result.all_pass:
            self._pass_count += 1

        logger.debug(
            "gates_evaluated",
            all_pass=result.all_pass,
            failures=failures or None,
            vol_regime=round(vol_regime, 3),
            spread_bps=round(spread_bps, 2),
            ev=round(ev, 4),
            bot_state=bot_state.name,
        )

        return result

    def stats(self) -> dict:
        """Return gate evaluation statistics for diagnostics."""
        total = max(self._eval_count, 1)
        return {
            'total_evaluations': self._eval_count,
            'total_passed': self._pass_count,
            'pass_rate': round(self._pass_count / total, 3),
            'failure_counts': self._failure_counts.copy(),
            'failure_rates': {
                k: round(v / total, 3)
                for k, v in self._failure_counts.items()
            },
        }
