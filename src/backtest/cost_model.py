"""
Cost Model — Polymarket fee and slippage simulation.

PRD 11.2:
  POLYMARKET_FEE = 0.02  (2% on gross profit)
  SLIPPAGE_BPS   = 5     (0.05% on bet size)
"""

from __future__ import annotations

POLYMARKET_FEE: float = 0.02    # 2% on gross profit
SLIPPAGE_BPS: int = 5           # 5 basis points


def net_payout(bet_size: float, odds: float, won: bool) -> float:
    """
    Calculate net PnL after fees and slippage.

    Args:
        bet_size: Amount wagered ($).
        odds:     Decimal odds (e.g., 1.90 means 90% profit on win).
        won:      Whether the bet was won.

    Returns:
        Net PnL (positive = profit, negative = loss).

    Examples:
        >>> net_payout(1.0, 1.90, True)   # win
        0.882  # gross 0.90 - 2% fee = 0.882
        >>> net_payout(1.0, 1.90, False)  # loss
        -1.0005  # bet + slippage
    """
    slippage = bet_size * (SLIPPAGE_BPS / 10_000)

    if won:
        gross_profit = bet_size * (odds - 1.0)
        fee = gross_profit * POLYMARKET_FEE
        return gross_profit - fee - slippage
    else:
        return -(bet_size + slippage)


def effective_odds(odds: float) -> float:
    """Odds after fee deduction (what you effectively receive)."""
    raw_profit_pct = odds - 1.0
    after_fee = raw_profit_pct * (1.0 - POLYMARKET_FEE)
    return 1.0 + after_fee


def break_even_win_rate(odds: float) -> float:
    """Minimum win rate to break even at given odds after fees."""
    eff = effective_odds(odds)
    # breakeven: p * (eff - 1) = (1 - p) * 1
    # p * eff = 1  =>  p = 1 / eff
    return 1.0 / eff
