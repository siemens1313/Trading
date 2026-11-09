"""Causal indicators: the value on day t uses data up to and including day t only.

Every function takes a 1-D series or a (days, symbols) array and returns the same shape, with NaN
until enough history exists.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _rolling(x: np.ndarray, n: int, reduce) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if 0 < n <= x.shape[0]:
        out[n - 1 :] = reduce(sliding_window_view(x, n, axis=0), axis=-1)
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling(x, n, np.mean)


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling(x, n, lambda w, axis: np.std(w, axis=axis, ddof=1))


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling(x, n, np.max)


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    return _rolling(x, n, np.min)


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential average seeded with the first finite value; a NaN input keeps the last value."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    prev = np.full(x.shape[1:], np.nan)
    for t in range(len(x)):
        v = x[t]
        prev = np.where(np.isnan(v), prev, np.where(np.isnan(prev), v, prev + alpha * (v - prev)))
        out[t] = prev
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's relative strength index, 0-100."""
    c = np.asarray(close, dtype=float)
    d = np.diff(c, axis=0, prepend=np.nan)
    gain = ema(np.where(np.isnan(d), np.nan, np.maximum(d, 0)), 1 / n)
    loss = ema(np.where(np.isnan(d), np.nan, np.maximum(-d, 0)), 1 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(loss > 0, 100 - 100 / (1 + gain / loss), np.where(gain > 0, 100.0, 50.0))
    out[: n + 1] = np.nan
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.vstack([np.full_like(close[:1], np.nan), close[:-1]]) if close.ndim > 1 else np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    out = ema(tr, 1 / n)
    out[:n] = np.nan
    return out


def bollinger(close: np.ndarray, n: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Middle, upper and lower bands, and bandwidth (upper - lower) / middle."""
    mid = sma(close, n)
    sd = _rolling(close, n, np.std)
    upper, lower = mid + k * sd, mid - k * sd
    return mid, upper, lower, (upper - lower) / mid


def efficiency_ratio(close: np.ndarray, n: int = 10) -> np.ndarray:
    """Kaufman's efficiency ratio: net move / path length over n days. Near 1 = trending, near 0 = choppy."""
    c = np.asarray(close, dtype=float)
    out = np.full(c.shape, np.nan)
    if n < len(c):
        net = np.abs(c[n:] - c[:-n])
        path = _rolling(np.abs(np.diff(c, axis=0)), n, np.sum)[n - 1 :]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n:] = np.where(path > 0, net / path, 0.0)
    return out


def kama(close: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman's adaptive moving average: fast in trends, nearly flat in chop."""
    c = np.asarray(close, dtype=float)
    er = efficiency_ratio(c, n)
    sc = (np.nan_to_num(er) * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(c.shape, np.nan)
    if len(c) > n:
        out[n] = c[n]
        for t in range(n + 1, len(c)):
            out[t] = out[t - 1] + sc[t] * (c[t] - out[t - 1])
    return out


def zigzag(series: np.ndarray, threshold: float) -> list[tuple[int, float, int, int]]:
    """Swing pivots of a 1-D series (use log prices): (index, value, kind, confirmed_at).

    kind is +1 for a swing high and -1 for a swing low. A pivot is confirmed only once the series has
    moved `threshold` away from it, on day `confirmed_at`, so it may be used from that day on.
    """
    pivots: list[tuple[int, float, int, int]] = []
    hi = lo = 0
    direction = 0
    for t in range(1, len(series)):
        v = series[t]
        if direction == 0:
            hi = t if v > series[hi] else hi
            lo = t if v < series[lo] else lo
            if v - series[lo] >= threshold:
                pivots.append((lo, float(series[lo]), -1, t))
                direction, hi = 1, t
            elif series[hi] - v >= threshold:
                pivots.append((hi, float(series[hi]), 1, t))
                direction, lo = -1, t
        elif direction == 1:
            if v > series[hi]:
                hi = t
            elif series[hi] - v >= threshold:
                pivots.append((hi, float(series[hi]), 1, t))
                direction, lo = -1, t
        else:
            if v < series[lo]:
                lo = t
            elif v - series[lo] >= threshold:
                pivots.append((lo, float(series[lo]), -1, t))
                direction, hi = 1, t
    return pivots
