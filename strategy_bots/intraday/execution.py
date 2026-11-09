"""Strategies 6 and 7: TWAP, VWAP and implementation-shortfall execution.

A parent order is cut into child orders over a window of a session's bars:
- TWAP: equal slices, whatever the volume.
- VWAP: slices in proportion to each bar's average share of the day's volume in past sessions.
- Implementation shortfall (Almgren-Chriss): front-loaded. Urgency 0 is TWAP; higher urgency trades
  sooner, cutting the risk of the price running away at the price of more market impact.
Fills are simulated at each bar's typical price (high + low + close) / 3, made worse by a half-spread
plus square-root market impact: impact_bps * sqrt(child size / bar volume).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..data import Session


def allocate(qty: int, weights: np.ndarray) -> np.ndarray:
    """Whole shares in proportion to weights, summing exactly to qty (largest remainder)."""
    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    if qty <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=int)
    raw = qty * w / w.sum()
    out = np.floor(raw).astype(int)
    out[np.argsort(-(raw - out), kind="stable")[: qty - out.sum()]] += 1
    return out


def twap(qty: int, bars: int) -> np.ndarray:
    return allocate(qty, np.ones(bars))


def vwap(qty: int, profile: np.ndarray) -> np.ndarray:
    return allocate(qty, profile)


def implementation_shortfall(qty: int, bars: int, urgency: float) -> np.ndarray:
    """Almgren-Chriss trajectory: holdings fall as sinh(u * (1 - j/n)) / sinh(u)."""
    j = np.arange(bars + 1) / bars
    holding = np.sinh(urgency * (1 - j)) / math.sinh(urgency) if urgency > 0 else 1 - j
    return allocate(qty, -np.diff(holding))


def volume_profile(sessions: list[Session], bars: int | None = None) -> np.ndarray:
    """Average share of the day's volume traded in each bar, over sessions with the full bar count."""
    bars = bars or max(len(s) for s in sessions)
    shares = [s.volume / s.volume.sum() for s in sessions if len(s) == bars and s.volume.sum() > 0]
    if not shares:
        raise ValueError(f"no complete sessions with {bars} bars to build a volume profile")
    profile = np.mean(shares, axis=0)
    return profile / profile.sum()


@dataclass
class ExecutionReport:
    algo: str
    side: str
    qty: int
    avg_price: float
    arrival: float
    interval_vwap: float
    cost_vs_arrival_bps: float  # positive = paid more (buy) or received less (sell) than the arrival price
    cost_vs_vwap_bps: float
    max_participation: float
    children: list[tuple[int, int, float]] = field(default_factory=list)  # (minute, shares, price)


def simulate(
    session: Session,
    schedule: np.ndarray,
    start: int = 0,
    side: str = "buy",
    algo: str = "",
    half_spread_bps: float = 1.0,
    impact_bps: float = 10.0,
) -> ExecutionReport:
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if start + len(schedule) > len(session):
        raise ValueError("the schedule runs past the end of the session")
    sign = 1 if side == "buy" else -1
    span = slice(start, start + len(schedule))
    typical = (session.high + session.low + session.close) / 3
    children, participation = [], 0.0
    for k, shares in enumerate(schedule):
        if shares <= 0:
            continue
        bar = start + k
        part = shares / max(session.volume[bar], 1.0)
        participation = max(participation, part)
        price = typical[bar] * (1 + sign * (half_spread_bps + impact_bps * math.sqrt(part)) / 1e4)
        children.append((int(session.minutes[bar]), int(shares), float(price)))
    qty = int(np.sum(schedule))
    avg = sum(s * p for _, s, p in children) / qty if qty else math.nan
    arrival = float(session.open[start])
    window_vwap = float(np.sum(typical[span] * session.volume[span]) / np.sum(session.volume[span]))
    return ExecutionReport(
        algo, side, qty, avg, arrival, window_vwap,
        sign * (avg / arrival - 1) * 1e4, sign * (avg / window_vwap - 1) * 1e4, participation, children,
    )
