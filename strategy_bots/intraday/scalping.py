"""Strategy 2: scalping.

Real tape reading needs every trade and quote; this reads the tape from bars instead. Order-flow
imbalance is the volume of up bars minus the volume of down bars over the last few bars, as a share of
their total. When buyers dominate and price takes out the previous bar's high, buy at the next bar's open.
Exit at a small profit target or a tighter stop (when a bar touches both, assume the stop), after
max_hold_bars, or at the close. Every trade pays a half-spread on the way in and out.

The pattern-day-trader rule: a margin account under $25,000 may make at most 3 day trades in any 5
business days. Scalping is nothing but day trades, so past that limit the bot refuses new entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..data import Session


@dataclass
class Trade:
    day: date
    entry_minute: int
    exit_minute: int
    entry: float
    exit: float
    return_bps: float
    reason: str


@dataclass
class ScalpResult:
    trades: list[Trade] = field(default_factory=list)
    blocked_by_pdt: int = 0

    def metrics(self) -> dict[str, float]:
        r = np.array([t.return_bps for t in self.trades])
        if not len(r):
            return {"trades": 0, "blocked_by_pdt": self.blocked_by_pdt}
        return {
            "trades": len(r),
            "win_rate": float((r > 0).mean()),
            "avg_bps": float(r.mean()),
            "total_bps": float(r.sum()),
            "blocked_by_pdt": self.blocked_by_pdt,
        }


class Scalper:
    defaults = {
        "imbalance_bars": 3,
        "entry_imbalance": 0.6,
        "take_profit_bps": 15.0,
        "stop_bps": 10.0,
        "max_hold_bars": 6,
        "half_spread_bps": 1.0,
        "pdt_equity": 25_000.0,
        "pdt_max_day_trades": 3,
    }

    def __init__(self, **params):
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ValueError(f"scalping: unknown parameter(s): {', '.join(unknown)}")
        self.p = {**self.defaults, **params}

    def run(self, sessions: list[Session], equity: float) -> ScalpResult:
        p = self.p
        result = ScalpResult()
        cost = p["half_spread_bps"] / 1e4
        n = p["imbalance_bars"]
        for s in sessions:
            k = n
            while k < len(s) - 1:
                window = slice(k - n + 1, k + 1)
                signed = np.sign(s.close[window] - s.open[window]) * s.volume[window]
                imbalance = signed.sum() / max(s.volume[window].sum(), 1.0)
                if not (imbalance >= p["entry_imbalance"] and s.close[k] > s.high[k - 1]):
                    k += 1
                    continue
                if equity < p["pdt_equity"] and self._recent_day_trades(result.trades, s.day) >= p["pdt_max_day_trades"]:
                    result.blocked_by_pdt += 1  # a signal the pattern-day-trader rule refused
                    k += 1
                    continue
                trade, k = self._hold(s, k + 1, cost)
                result.trades.append(trade)
        return result

    def _hold(self, s: Session, entry_bar: int, cost: float) -> tuple[Trade, int]:
        p = self.p
        entry = s.open[entry_bar] * (1 + cost)
        target = entry * (1 + p["take_profit_bps"] / 1e4)
        stop = entry * (1 - p["stop_bps"] / 1e4)
        last = min(entry_bar + p["max_hold_bars"] - 1, len(s) - 1)
        for j in range(entry_bar, last + 1):
            if s.low[j] <= stop:
                price, reason = stop, "stop"
            elif s.high[j] >= target:
                price, reason = target, "target"
            elif j == last:
                price, reason = s.close[j], "close" if j == len(s) - 1 else "time"
            else:
                continue
            exit_ = price * (1 - cost)
            return Trade(s.day, int(s.minutes[entry_bar]), int(s.minutes[j]), entry, exit_, (exit_ / entry - 1) * 1e4, reason), j + 1
        raise AssertionError("unreachable: the last bar always exits")

    @staticmethod
    def _recent_day_trades(trades: list[Trade], today: date) -> int:
        """Day trades in the 5 business days ending today."""
        today64 = np.datetime64(today, "D")
        return sum(1 for t in trades if 0 <= np.busday_count(np.datetime64(t.day, "D"), today64) < 5)
