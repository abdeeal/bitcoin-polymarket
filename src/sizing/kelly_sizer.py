"""
Kelly Sizer — Fractional Kelly bet sizing for balance >= $100.

Formula (PRD 8.4):
  k   = (b × p − q) / b
  bet = min(k × 0.25 × balance, balance × 0.02)

where:
  b = decimal_odds − 1   (net profit per unit wagered)
  p = model_prob          (probability of winning)
  q = 1 − p              (probability of losing)
  0.25 = Kelly fraction   (25% of full Kelly — reduces variance)
  0.02 = max bet cap      (never risk > 2% of balance per trade)

Why fractional Kelly (0.25)?
  Full Kelly maximises log-expected utility but produces extreme variance.
  25% Kelly retains most of the growth benefit while reducing drawdown
  by ~75%. This is standard practice in quantitative trading.

Why 2% hard cap?
  Protects against model over-confidence. Even if Kelly suggests 15%,
  we never risk more than 2% per trade. At $100 balance, max bet = $2.00.

Edge cases:
  - Kelly < 0 (negative edge): returns 0.0 — never bet on negative EV
  - Odds <= 1: invalid (negative profit), returns 0.0
  - p = 0 or p = 1: clipped to avoid numerical issues
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KellyResult:
    """Detailed Kelly computation output for logging."""
    p: float             # model probability
    q: float             # 1 - p
    b: float             # odds - 1 (net profit ratio)
    raw_kelly: float     # full Kelly fraction
    fractional_kelly: float   # after applying Kelly fraction
    bet_size: float      # final dollar amount (after cap)
    balance: float
    fraction_used: float      # fraction of balance wagered
    capped: bool         # True if max_pct cap was binding

    def to_dict(self) -> dict:
        return {
            'p': round(self.p, 4),
            'b': round(self.b, 4),
            'raw_kelly': round(self.raw_kelly, 4),
            'fractional_kelly': round(self.fractional_kelly, 4),
            'bet_size': round(self.bet_size, 2),
            'balance': round(self.balance, 2),
            'fraction_of_balance': round(self.fraction_used, 4),
            'capped_at_max_pct': self.capped,
        }


class KellySizer:
    """
    Fractional Kelly position sizer.

    Used exclusively when CompoundingEngine.balance >= $100.
    Below $100, fixed bracket bet sizes are used instead.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        max_pct: float = 0.02,
    ):
        """
        Args:
            fraction: Kelly multiplier (0.25 = 25% Kelly). PRD default: 0.25.
            max_pct:  Maximum fraction of balance per bet. PRD default: 0.02 (2%).
        """
        if not (0 < fraction <= 1):
            raise ValueError(f"Kelly fraction must be in (0, 1], got {fraction}")
        if not (0 < max_pct <= 0.10):
            raise ValueError(f"max_pct must be in (0, 0.10], got {max_pct}")

        self._fraction = fraction
        self._max_pct = max_pct

    def size(
        self,
        prob: float,
        odds: float,
        balance: float,
    ) -> float:
        """
        Compute fractional Kelly bet size in dollars.

        Args:
            prob:    Probability of winning this bet (P(correct direction)).
            odds:    Decimal odds (e.g., 1.90 = 90% profit on a win).
            balance: Current wallet balance.

        Returns:
            Dollar bet amount, rounded to 2 decimal places.
            Returns 0.0 for negative-EV scenarios (raw_kelly <= 0).
        """
        result = self.compute(prob, odds, balance)
        return result.bet_size

    def compute(
        self,
        prob: float,
        odds: float,
        balance: float,
    ) -> KellyResult:
        """
        Compute full Kelly details (for diagnostics and logging).

        Returns KellyResult with all intermediate values.
        """
        # Clip probability to avoid numerical edge cases
        p = max(0.001, min(0.999, prob))
        q = 1.0 - p

        # Validate odds
        if odds <= 1.0:
            logger.warning("kelly_invalid_odds", odds=odds)
            return KellyResult(p, q, 0.0, 0.0, 0.0, 0.0, balance, 0.0, False)

        b = odds - 1.0  # net profit per unit wagered

        # Raw Kelly fraction: (b*p - q) / b
        raw_kelly = (b * p - q) / b

        # Negative Kelly → no edge → no bet
        if raw_kelly <= 0:
            logger.debug("kelly_negative_edge", raw_kelly=round(raw_kelly, 4), p=p, b=b)
            return KellyResult(p, q, b, raw_kelly, 0.0, 0.0, balance, 0.0, False)

        # Apply Kelly fraction (25%)
        fractional = raw_kelly * self._fraction

        # Dollar amount from fractional Kelly
        kelly_bet = fractional * balance

        # Apply max_pct cap (2% of balance)
        max_bet = self._max_pct * balance
        capped = kelly_bet > max_bet
        final_bet = min(kelly_bet, max_bet)
        final_bet = max(0.0, min(final_bet, balance))
        final_bet = round(final_bet, 2)

        fraction_used = final_bet / balance if balance > 0 else 0.0

        result = KellyResult(
            p=p, q=q, b=b,
            raw_kelly=raw_kelly,
            fractional_kelly=fractional,
            bet_size=final_bet,
            balance=balance,
            fraction_used=fraction_used,
            capped=capped,
        )

        logger.debug(
            "kelly_computed",
            p=round(p, 4),
            b=round(b, 4),
            raw_kelly=round(raw_kelly, 4),
            bet=final_bet,
            capped=capped,
        )

        return result

    def expected_growth_rate(self, prob: float, odds: float) -> float:
        """
        Compute expected log-growth rate for fractional Kelly.

        Used for diagnostics — positive value means the strategy
        is expected to grow capital over the long run.

        Returns: E[log(1 + f×b)] × N (annualised approximation)
        """
        p = max(0.001, min(0.999, prob))
        q = 1.0 - p
        if odds <= 1.0:
            return 0.0
        b = odds - 1.0
        f = max(0.0, (b * p - q) / b) * self._fraction
        win_growth = math.log(1.0 + f * b) if (1.0 + f * b) > 0 else 0.0
        lose_growth = math.log(1.0 - f) if (1.0 - f) > 0 else -float('inf')
        return p * win_growth + q * lose_growth

    @property
    def fraction(self) -> float:
        return self._fraction

    @property
    def max_pct(self) -> float:
        return self._max_pct
