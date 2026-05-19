"""
Numba JIT Kernels — Custom hot-path computations compiled to machine code.

All kernels use @njit(cache=True, fastmath=True):
  - cache=True  → compiled result cached to disk, instant load on subsequent runs
  - fastmath=True → allow IEEE 754 relaxations for ~2x speedup on floating point

These kernels handle computations that cannot be expressed as pure NumPy
vectorized operations (rolling windows with conditional logic, sequential
accumulation, etc.) while still running at near-C speed.

PRD RULE: All @njit kernels MUST be pre-compiled at boot via a dummy call.
This eliminates first-call JIT overhead (typically 200ms-2s).

Kernels in this file:
  1. compute_ofi          — Order Flow Imbalance
  2. compute_kyle_lambda  — Kyle's Lambda (market impact, rolling)
  3. compute_amihud       — Amihud illiquidity ratio (rolling)
  4. compute_vwap_rolling — Volume-Weighted Average Price (rolling)
  5. percentile_rank_rolling — Rolling percentile rank
  6. compute_consecutive_count — Consecutive green/red candle counter
  7. compute_rolling_correlation — Rolling Pearson correlation
  8. compute_volume_delta_cumulative — Cumulative volume delta
"""

from numba import njit
import numpy as np


# =============================================================================
# 1. Order Flow Imbalance (OFI)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_ofi(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
) -> float:
    """
    Compute Order Flow Imbalance from order book snapshots.

    OFI measures the net change in bid/ask depth, accounting for
    price level changes. Positive OFI = buying pressure, negative = selling.

    The calculation follows:
      - If bid price unchanged: delta_bid = current_size - previous_size
      - If bid price changed: delta_bid = current_size (new level appeared)
      - Same logic for asks (inverted sign)

    Args:
        bid_sizes:  Array of best bid sizes over time.
        ask_sizes:  Array of best ask sizes over time.
        bid_prices: Array of best bid prices over time.
        ask_prices: Array of best ask prices over time.

    Returns:
        Scalar OFI value (positive = buy pressure, negative = sell pressure).
    """
    ofi = 0.0
    n = len(bid_sizes)
    for i in range(1, n):
        # Bid side contribution
        if bid_prices[i] == bid_prices[i - 1]:
            delta_bid = bid_sizes[i] - bid_sizes[i - 1]
        elif bid_prices[i] > bid_prices[i - 1]:
            delta_bid = bid_sizes[i]
        else:
            delta_bid = -bid_sizes[i - 1]

        # Ask side contribution
        if ask_prices[i] == ask_prices[i - 1]:
            delta_ask = ask_sizes[i] - ask_sizes[i - 1]
        elif ask_prices[i] < ask_prices[i - 1]:
            delta_ask = ask_sizes[i]
        else:
            delta_ask = -ask_sizes[i - 1]

        ofi += delta_bid - delta_ask

    return ofi


@njit(cache=True, fastmath=True)
def compute_ofi_rolling(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling OFI over a sliding window.

    Args:
        bid_sizes:  Array of best bid sizes over time.
        ask_sizes:  Array of best ask sizes over time.
        bid_prices: Array of best bid prices over time.
        ask_prices: Array of best ask prices over time.
        window:     Rolling window size.

    Returns:
        Array of rolling OFI values. First (window) entries are NaN.
    """
    n = len(bid_sizes)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan

    for i in range(window, n):
        ofi = 0.0
        for j in range(i - window + 1, i + 1):
            if j == 0:
                continue
            if bid_prices[j] == bid_prices[j - 1]:
                delta_bid = bid_sizes[j] - bid_sizes[j - 1]
            elif bid_prices[j] > bid_prices[j - 1]:
                delta_bid = bid_sizes[j]
            else:
                delta_bid = -bid_sizes[j - 1]

            if ask_prices[j] == ask_prices[j - 1]:
                delta_ask = ask_sizes[j] - ask_sizes[j - 1]
            elif ask_prices[j] < ask_prices[j - 1]:
                delta_ask = ask_sizes[j]
            else:
                delta_ask = -ask_sizes[j - 1]

            ofi += delta_bid - delta_ask
        result[i] = ofi

    return result


# =============================================================================
# 2. Kyle's Lambda (Market Impact)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_kyle_lambda(
    returns: np.ndarray,
    volumes: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling Kyle's Lambda — measures price impact per unit of volume.

    Lambda = mean(|return| / sqrt(volume)) over a rolling window.
    Higher values indicate lower liquidity (more price impact per trade).

    Args:
        returns: Array of price returns.
        volumes: Array of trade volumes.
        window:  Rolling window size.

    Returns:
        Array of Kyle's Lambda values. First (window) entries are NaN.
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan

    for i in range(window, n):
        total = 0.0
        count = 0
        for j in range(i - window, i):
            sv = np.sqrt(volumes[j])
            if sv < 1e-10:
                sv = 1e-10
            total += np.abs(returns[j]) / sv
            count += 1
        result[i] = total / count if count > 0 else 0.0

    return result


# =============================================================================
# 3. Amihud Illiquidity Ratio
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_amihud(
    returns: np.ndarray,
    volumes: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling Amihud illiquidity ratio.

    Amihud = mean(|return| / volume) over a rolling window.
    Higher values indicate lower liquidity.

    Args:
        returns: Array of price returns.
        volumes: Array of trade volumes (in quote currency).
        window:  Rolling window size.

    Returns:
        Array of Amihud ratios. First (window) entries are NaN.
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan

    for i in range(window, n):
        total = 0.0
        count = 0
        for j in range(i - window, i):
            vol = volumes[j]
            if vol < 1e-10:
                vol = 1e-10
            total += np.abs(returns[j]) / vol
            count += 1
        result[i] = total / count if count > 0 else 0.0

    return result


# =============================================================================
# 4. Rolling VWAP (Volume-Weighted Average Price)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_vwap_rolling(
    close: np.ndarray,
    volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling VWAP over a sliding window.

    VWAP = sum(price × volume) / sum(volume) over the window.

    Args:
        close:  Array of close prices.
        volume: Array of trade volumes.
        window: Rolling window size.

    Returns:
        Array of VWAP values. First (window-1) entries are NaN.
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        pv_sum = 0.0
        vol_sum = 0.0
        for j in range(i - window + 1, i + 1):
            pv_sum += close[j] * volume[j]
            vol_sum += volume[j]
        result[i] = pv_sum / (vol_sum + 1e-10)

    return result


# =============================================================================
# 5. Rolling Percentile Rank
# =============================================================================

@njit(cache=True, fastmath=True)
def percentile_rank_rolling(
    arr: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling percentile rank of the current value within its window.

    Rank = count(values <= current) / window_size.
    Range: [0, 1] where 1 means current value is the maximum in the window.

    Args:
        arr:    Input array.
        window: Rolling window size.

    Returns:
        Array of percentile ranks. First (window) entries are NaN.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan

    for i in range(window, n):
        count = 0
        for j in range(i - window, i):
            if arr[j] <= arr[i]:
                count += 1
        result[i] = count / window

    return result


# =============================================================================
# 6. Consecutive Count (Green/Red candles)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_consecutive_count(
    directions: np.ndarray,
    target: float,
) -> np.ndarray:
    """
    Count consecutive occurrences of a target direction.

    Used for: consecutive_green (target=1) and consecutive_red (target=-1).

    Args:
        directions: Array of candle directions (+1 green, -1 red, 0 doji).
        target:     Direction to count (1.0 for green, -1.0 for red).

    Returns:
        Array where each element is the count of consecutive target values
        ending at that position.
    """
    n = len(directions)
    result = np.empty(n, dtype=np.float64)

    count = 0.0
    for i in range(n):
        if directions[i] == target:
            count += 1.0
        else:
            count = 0.0
        result[i] = count

    return result


# =============================================================================
# 7. Rolling Pearson Correlation
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_rolling_correlation(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling Pearson correlation coefficient between two arrays.

    Args:
        x:      First input array.
        y:      Second input array.
        window: Rolling window size.

    Returns:
        Array of correlation values [-1, 1]. First (window) entries are NaN.
    """
    n = len(x)
    result = np.empty(n, dtype=np.float64)
    result[:window] = np.nan

    for i in range(window, n):
        x_w = x[i - window:i]
        y_w = y[i - window:i]

        x_mean = 0.0
        y_mean = 0.0
        for j in range(window):
            x_mean += x_w[j]
            y_mean += y_w[j]
        x_mean /= window
        y_mean /= window

        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for j in range(window):
            dx = x_w[j] - x_mean
            dy = y_w[j] - y_mean
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        denom = np.sqrt(var_x * var_y)
        if denom < 1e-10:
            result[i] = 0.0
        else:
            result[i] = cov / denom

    return result


# =============================================================================
# 8. Cumulative Volume Delta
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_volume_delta_cumulative(
    taker_buy: np.ndarray,
    total_volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Cumulative volume delta over a rolling window.

    Delta = taker_buy_volume - taker_sell_volume
    where taker_sell = total_volume - taker_buy.

    Positive delta = net buying; negative = net selling.

    Args:
        taker_buy:    Array of taker buy volumes.
        total_volume: Array of total volumes.
        window:       Rolling window size.

    Returns:
        Array of cumulative delta values. First (window-1) entries are NaN.
    """
    n = len(taker_buy)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        cum_delta = 0.0
        for j in range(i - window + 1, i + 1):
            buy = taker_buy[j]
            sell = total_volume[j] - buy
            cum_delta += buy - sell
        result[i] = cum_delta

    return result


# =============================================================================
# 9. Rolling Mean (simple, for normalization)
# =============================================================================

@njit(cache=True, fastmath=True)
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Simple rolling mean using cumulative sum trick.

    Args:
        arr:    Input array.
        window: Rolling window size.

    Returns:
        Array of rolling means. First (window-1) entries are NaN.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan

    # Initial window sum
    running_sum = 0.0
    for j in range(window):
        running_sum += arr[j]
    result[window - 1] = running_sum / window

    # Slide the window
    for i in range(window, n):
        running_sum += arr[i] - arr[i - window]
        result[i] = running_sum / window

    return result


# =============================================================================
# 10. Rolling Standard Deviation
# =============================================================================

@njit(cache=True, fastmath=True)
def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation using Welford's online algorithm.

    Args:
        arr:    Input array.
        window: Rolling window size.

    Returns:
        Array of rolling std values. First (window-1) entries are NaN.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += arr[j]
        mean /= window

        var = 0.0
        for j in range(i - window + 1, i + 1):
            diff = arr[j] - mean
            var += diff * diff
        var /= window

        result[i] = np.sqrt(var)

    return result


# =============================================================================
# Pre-compilation Helper
# =============================================================================

def precompile_all_kernels() -> None:
    """
    Pre-compile all JIT kernels by calling each with a small dummy array.

    MUST be called at boot before the live trading loop begins.
    Eliminates first-call JIT compilation latency (200ms-2s per kernel).
    """
    dummy = np.random.rand(50).astype(np.float64)
    dummy_int = np.random.randint(0, 100, 50).astype(np.float64)

    # Core kernels
    compute_ofi(dummy, dummy, dummy, dummy)
    compute_ofi_rolling(dummy, dummy, dummy, dummy, 10)
    compute_kyle_lambda(dummy, dummy, 10)
    compute_amihud(dummy, dummy, 10)
    compute_vwap_rolling(dummy, dummy, 10)
    percentile_rank_rolling(dummy, 10)

    # Utility kernels
    directions = np.sign(np.random.randn(50)).astype(np.float64)
    compute_consecutive_count(directions, 1.0)
    compute_rolling_correlation(dummy, dummy, 10)
    compute_volume_delta_cumulative(dummy, dummy * 2, 10)
    rolling_mean(dummy, 10)
    rolling_std(dummy, 10)
