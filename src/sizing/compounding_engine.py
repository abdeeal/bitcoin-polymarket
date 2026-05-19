"""
Compounding Engine — Fixed Bracket System + Kelly Sizing at $100+.

Bracket Table:
  ┌───────┬──────────┬──────────┬────────┬───────────┐
  │ Index │  Floor   │  Ceil    │ Normal │ High-Conf │
  ├───────┼──────────┼──────────┼────────┼───────────┤
  │   0   │  $5.00   │ $14.99   │ $1.00  │   $2.00   │
  │   1   │ $15.00   │ $24.99   │ $1.50  │   $3.00   │
  │   2   │ $25.00   │ $39.99   │ $2.50  │   $5.00   │
  │   3   │ $40.00   │ $64.99   │ $4.00  │   $8.00   │
  │   4   │ $65.00   │ $99.99   │ $6.50  │  $13.00   │
  │   5   │ $100.00  │    ∞     │ Kelly  │   Kelly   │
  └───────┴──────────┴──────────┴────────┴───────────┘

Upgrade Rules:
  balance >= bracket.ceil
  AND bets_in_bracket >= 10   (minimum sample size)
  AND win_rate_in_bracket >= 0.50   (positive record required)
  → If win_rate < 50% at ceiling: defer upgrade, log warning

Downgrade Rules:
  balance < bracket.floor → immediate downgrade to matching bracket
  Resets bets_in_bracket and win_rate_in_bracket to zero

No Hard Stop:
  Betting continues until balance is genuinely insufficient to cover
  the minimum available bet ($1.00 normal). The bot never forces
  a PAUSED state due to low balance — it simply cannot place a bet
  when resolve_bet_size() returns 0.0 (balance < bet size).
  BotState.PAUSED is reserved for cooling-down expiry and manual stop.

Kelly Mode (balance >= $100):
  k  = (b × p − q) / b   (raw Kelly fraction)
  bet = min(k × 0.25 × balance, balance × 0.02)
  where b = odds − 1, p = model_prob, q = 1 − p
  Fractional Kelly (0.25): reduces variance
  Max cap (2% of balance): hard ceiling
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import structlog

from src.signal.signal_generator import Signal, BotState

logger = structlog.get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

KELLY_MODE_THRESHOLD: float = 100.00  # balance >= this → Kelly sizing
KELLY_FRACTION: float = 0.25          # fractional Kelly multiplier (25%)
KELLY_MAX_PCT: float = 0.02           # max 2% of balance per bet
UPGRADE_MIN_BETS: int = 10            # minimum bets in bracket before upgrade
UPGRADE_MIN_WIN_RATE: float = 0.50    # minimum win rate for upgrade
MINIMUM_BET_FLOOR: float = 1.00       # smallest normal bet — bot stops only below this


# ─── Bracket definition ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bracket:
    """A single balance bracket with associated bet sizes."""
    index: int
    floor: float       # minimum balance for this bracket (inclusive)
    ceil: float        # maximum balance — upgrade triggers at >= ceil
    normal_bet: Optional[float]     # None for Kelly mode (bracket 5)
    high_conf_bet: Optional[float]  # None for Kelly mode (bracket 5)

    @property
    def is_kelly_mode(self) -> bool:
        return self.normal_bet is None

    def __str__(self) -> str:
        if self.is_kelly_mode:
            return f"Bracket[{self.index}] ${self.floor:.2f}+ (Kelly mode)"
        return (
            f"Bracket[{self.index}] ${self.floor:.2f}–${self.ceil:.2f} "
            f"(normal=${self.normal_bet:.2f}, high_conf=${self.high_conf_bet:.2f})"
        )


# PRD 8.1 — complete bracket table
BRACKETS: list[Bracket] = [
    Bracket(0,   5.00,  14.99,  1.00,  2.00),
    Bracket(1,  15.00,  24.99,  1.50,  3.00),
    Bracket(2,  25.00,  39.99,  2.50,  5.00),
    Bracket(3,  40.00,  64.99,  4.00,  8.00),
    Bracket(4,  65.00,  99.99,  6.50, 13.00),
    Bracket(5, 100.00, math.inf, None, None),   # Kelly mode
]


# ─── Compounding Engine ───────────────────────────────────────────────────────

class CompoundingEngine:
    """
    Fixed bracket compounding with Kelly mode at $100+.

    Tracks balance, bracket position, and win rate within bracket.
    Enforces upgrade/downgrade rules after every resolved bet.

    Stopping Policy (No Hard Stop):
      resolve_bet_size() returns 0.0 when balance < minimum bet.
      The calling loop interprets 0.0 as "cannot bet" and skips.
      BotState.PAUSED is never set automatically by this engine.

    Thread Safety:
      All state mutations happen synchronously in the I/O event loop.
      No locking required (single-writer pattern).
    """

    def __init__(
        self,
        initial_balance: float = 5.00,
        on_state_change: Optional[Callable[[BotState], None]] = None,
        on_alert: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            initial_balance:  Starting wallet balance.
            on_state_change:  Callback invoked when bot state changes
                              (e.g., to trigger cooling down or paused).
            on_alert:         Callback for hard-stop and bracket change alerts.
        """
        self._balance: float = initial_balance
        self._current_bracket: Bracket = self._find_bracket(initial_balance)
        self._bot_state: BotState = BotState.ACTIVE

        # Per-bracket statistics
        self._bets_in_bracket: int = 0
        self._wins_in_bracket: int = 0

        # Lifetime statistics
        self._total_bets: int = 0
        self._total_wins: int = 0
        self._total_wagered: float = 0.0
        self._total_pnl: float = 0.0
        self._peak_balance: float = initial_balance

        # Last signal context (needed for Kelly)
        self._last_prob: float = 0.5
        self._last_odds: float = 1.90

        # Callbacks
        self._on_state_change = on_state_change
        self._on_alert = on_alert

        logger.info(
            "compounding_engine_init",
            balance=initial_balance,
            bracket=str(self._current_bracket),
        )

    # ─── Bet Size Resolution ──────────────────────────────────────────────────

    def resolve_bet_size(
        self,
        signal: Signal,
        prob: float = 0.5,
        odds: float = 1.90,
    ) -> float:
        """
        Return the bet size in dollars for the given signal.

        Routing:
          - balance >= $100.00 → KellySizer (fractional Kelly)
          - otherwise → fixed bracket amount based on signal tier

        Stopping condition (no hard stop):
          Returns 0.0 when balance < the bracket's bet size.
          The caller interprets 0.0 as "cannot bet" and skips execution.
          BotState is never forced to PAUSED by this method.

        Args:
            signal: Signal enum (BET_UP_NORMAL, BET_UP_HIGH_CONF, etc.)
            prob:   Calibrated P(UP) from ensemble (for Kelly).
            odds:   Decimal odds from Polymarket (for Kelly).

        Returns:
            Bet size in dollars, rounded to 2 decimal places.
            Returns 0.0 if:
              - signal is NO_BET
              - bot_state is not ACTIVE
              - balance is insufficient to cover the bet
        """
        if not signal.is_bet or self._bot_state != BotState.ACTIVE:
            return 0.0

        self._last_prob = prob
        self._last_odds = odds

        if self._balance >= KELLY_MODE_THRESHOLD:
            return self._kelly_size(signal, prob, odds)

        bracket = self._current_bracket
        if signal.is_high_conf:
            bet = bracket.high_conf_bet or 0.0
        else:
            bet = bracket.normal_bet or 0.0

        # Balance insufficient to cover this bet → return 0.0 (no bet placed)
        if self._balance < bet:
            logger.info(
                "bet_skipped_insufficient_balance",
                balance=round(self._balance, 2),
                required=bet,
            )
            return 0.0

        return round(bet, 2)

    def _kelly_size(self, signal: Signal, prob: float, odds: float) -> float:
        """
        Fractional Kelly bet sizing for balance >= $100.

        Formula (PRD 8.4):
          k   = (b × p − q) / b
          bet = min(k × 0.25 × balance, balance × 0.02)

        where:
          b = odds − 1 (profit per unit wagered)
          p = model_prob (P winning)
          q = 1 − p    (P losing)

        The 0.25 fraction reduces the raw Kelly bet to 25%,
        dramatically reducing variance and risk of ruin.
        The 2% cap is a hard ceiling regardless of Kelly output.

        Args:
            signal: Signal direction (determines p vs 1-p).
            prob:   P(UP) from the model.
            odds:   Decimal odds (e.g., 1.90).

        Returns:
            Dollar bet size, rounded to 2 decimal places.
        """
        from src.sizing.kelly_sizer import KellySizer
        sizer = KellySizer(
            fraction=KELLY_FRACTION,
            max_pct=KELLY_MAX_PCT,
        )
        direction_prob = prob if signal.is_up else (1.0 - prob)
        return sizer.size(direction_prob, odds, self._balance)

    # ─── Bet Resolution ───────────────────────────────────────────────────────

    def on_bet_resolved(self, won: bool, bet_amount: float, payout: float = 0.0) -> dict:
        """
        Update state after a bet outcome is known.

        Called by the execution layer when Polymarket settles the market.
        Triggers upgrade/downgrade checks after every resolution.
        No hard stop — betting continues until balance is naturally
        insufficient (resolve_bet_size returns 0.0).

        Args:
            won:        True if bet was correct.
            bet_amount: Dollar amount wagered.
            payout:     Dollar amount received (0 if lost, > bet_amount if won).

        Returns:
            dict with updated state for logging.
        """
        # Update balance
        if won:
            profit = payout - bet_amount
            self._balance += profit
            self._total_wins += 1
            self._wins_in_bracket += 1
        else:
            self._balance -= bet_amount
            self._balance = max(self._balance, 0.0)

        # Update stats
        self._total_bets += 1
        self._bets_in_bracket += 1
        self._total_wagered += bet_amount
        self._total_pnl = self._balance - self._peak_balance

        if self._balance > self._peak_balance:
            self._peak_balance = self._balance

        # Downgrade if balance fell below bracket floor
        self._check_downgrade()

        # Upgrade if balance reached bracket ceiling
        if self._balance >= self._current_bracket.ceil:
            self._check_upgrade()

        logger.info(
            "bet_resolved",
            won=won,
            bet=round(bet_amount, 2),
            balance=round(self._balance, 2),
            bracket=self._current_bracket.index,
            bets_in_bracket=self._bets_in_bracket,
            win_rate=round(self.win_rate_in_bracket, 3),
            can_bet_normal=self._balance >= MINIMUM_BET_FLOOR,
        )

        return self._state_snapshot()

    # ─── Downgrade ────────────────────────────────────────────────────────────

    def _check_downgrade(self) -> None:
        """
        Immediate downgrade if balance falls below current bracket floor.

        Resets bets_in_bracket and wins_in_bracket to zero (fresh start
        in new bracket — previous record is irrelevant for new sizing).
        """
        if self._balance < self._current_bracket.floor:
            old_bracket = self._current_bracket
            self._current_bracket = self._find_bracket(self._balance)

            if self._current_bracket.index != old_bracket.index:
                self._bets_in_bracket = 0
                self._wins_in_bracket = 0
                msg = (
                    f"📉 BRACKET DOWNGRADE: "
                    f"[{old_bracket.index}] → [{self._current_bracket.index}] "
                    f"(balance=${self._balance:.2f})"
                )
                logger.warning("bracket_downgrade", old=old_bracket.index,
                               new=self._current_bracket.index, balance=self._balance)
                if self._on_alert:
                    self._on_alert(msg)

    # ─── Upgrade (PRD 8.2) ────────────────────────────────────────────────────

    def _check_upgrade(self) -> None:
        """
        Attempt bracket upgrade when balance reaches the ceiling.

        Conditions (ALL must be met):
          1. balance >= current_bracket.ceil
          2. bets_in_bracket >= 10
          3. win_rate_in_bracket >= 0.50

        If balance is at ceiling but win_rate < 50%:
          → Defer upgrade, log warning (protect against lucky streak at bad WR).
        """
        if not self.can_upgrade:
            # At ceiling but conditions not met
            if (self._balance >= self._current_bracket.ceil
                    and self._bets_in_bracket >= UPGRADE_MIN_BETS
                    and self.win_rate_in_bracket < UPGRADE_MIN_WIN_RATE):
                logger.warning(
                    "bracket_upgrade_deferred",
                    balance=self._balance,
                    bets=self._bets_in_bracket,
                    win_rate=round(self.win_rate_in_bracket, 3),
                    reason="win_rate_below_50pct",
                )
            return

        # Perform upgrade
        old_bracket = self._current_bracket
        next_idx = min(old_bracket.index + 1, len(BRACKETS) - 1)
        self._current_bracket = BRACKETS[next_idx]
        self._bets_in_bracket = 0
        self._wins_in_bracket = 0

        msg = (
            f"📈 BRACKET UPGRADE: "
            f"[{old_bracket.index}] → [{self._current_bracket.index}] "
            f"(balance=${self._balance:.2f}, "
            f"prev_wr={self.win_rate_in_bracket:.0%})"
        )
        logger.info(
            "bracket_upgrade",
            old=old_bracket.index,
            new=self._current_bracket.index,
            balance=self._balance,
        )
        if self._on_alert:
            self._on_alert(msg)

    @property
    def can_upgrade(self) -> bool:
        """All three upgrade conditions satisfied."""
        return (
            self._balance >= self._current_bracket.ceil
            and self._bets_in_bracket >= UPGRADE_MIN_BETS
            and self.win_rate_in_bracket >= UPGRADE_MIN_WIN_RATE
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_bracket(balance: float) -> Bracket:
        """Find the correct bracket for a given balance."""
        for bracket in reversed(BRACKETS):
            if balance >= bracket.floor:
                return bracket
        return BRACKETS[0]

    @property
    def win_rate_in_bracket(self) -> float:
        if self._bets_in_bracket == 0:
            return 0.0
        return self._wins_in_bracket / self._bets_in_bracket

    @property
    def overall_win_rate(self) -> float:
        if self._total_bets == 0:
            return 0.0
        return self._total_wins / self._total_bets

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def current_bracket(self) -> Bracket:
        return self._current_bracket

    @property
    def bot_state(self) -> BotState:
        return self._bot_state

    @property
    def is_kelly_mode(self) -> bool:
        return self._balance >= KELLY_MODE_THRESHOLD

    @property
    def can_bet(self) -> bool:
        """True when balance is sufficient to cover the minimum normal bet."""
        return self._balance >= MINIMUM_BET_FLOOR

    def _state_snapshot(self) -> dict:
        return {
            'balance': round(self._balance, 2),
            'bracket': self._current_bracket.index,
            'bracket_str': str(self._current_bracket),
            'bot_state': self._bot_state.name,
            'can_bet': self.can_bet,
            'bets_in_bracket': self._bets_in_bracket,
            'win_rate_in_bracket': round(self.win_rate_in_bracket, 3),
            'total_bets': self._total_bets,
            'overall_win_rate': round(self.overall_win_rate, 3),
            'total_pnl': round(self._total_pnl, 2),
            'peak_balance': round(self._peak_balance, 2),
            'is_kelly_mode': self.is_kelly_mode,
            'can_upgrade': self.can_upgrade,
        }

    def stats(self) -> dict:
        return self._state_snapshot()
