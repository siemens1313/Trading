"""Small helpers over chronological lists of values."""

from __future__ import annotations


def sessions_since_low(values: list[float]) -> int:
    """Sessions since the lowest value. A tie counts as the newer low (a re-test is a new low)."""
    low = min(values)
    last = max(i for i, v in enumerate(values) if v == low)
    return len(values) - 1 - last


def sma(values: list[float], n: int) -> float | None:
    if n <= 0 or len(values) < n:
        return None
    return sum(values[-n:]) / n


def percentile_rank(history: list[float], current: float) -> float:
    """Share of `history` strictly below `current`."""
    return sum(1 for v in history if v < current) / len(history)
