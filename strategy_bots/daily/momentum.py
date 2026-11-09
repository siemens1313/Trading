"""Strategy 5: momentum.

Every `rebalance` days, hold the `slots` names with the strongest return from `lookback` to `skip` days
ago, among names in an uptrend (close above the 200-day average) whose RSI confirms strength. A name is
dropped the day it loses its uptrend.
"""

from __future__ import annotations

import numpy as np

from ..indicators import rsi, sma
from .base import DailyStrategy, slots


class Momentum(DailyStrategy):
    name, number, title = "momentum", 5, "Momentum (relative strength + trend + RSI)"
    defaults = {"lookback": 126, "skip": 21, "slots": 5, "trend_sma": 200, "rsi_len": 14, "rsi_min": 50.0, "rebalance": 21}

    def warmup(self):
        return max(self.p["lookback"], self.p["trend_sma"], self.p["rsi_len"] + 1)

    def weights(self, panel):
        p = self.p
        c = panel.close
        T, N = c.shape
        L, S = p["lookback"], p["skip"]
        mom = np.full(c.shape, np.nan)
        if T > L:
            mom[L:] = c[L - S : T - S] / c[: T - L] - 1
        trend = c > sma(c, p["trend_sma"])
        eligible = trend & (rsi(c, p["rsi_len"]) > p["rsi_min"]) & (mom > 0)
        W = np.zeros((T, N))
        held = np.zeros(N)
        for t in range(T):
            if t % p["rebalance"] == 0:
                held = slots(eligible[t], p["slots"], mom[t])
            else:
                held = np.where(trend[t], held, 0.0)
            W[t] = held
        return W
