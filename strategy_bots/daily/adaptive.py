"""Strategy 14: adaptive regime switching.

Kaufman's efficiency ratio classifies each stock's regime every day. In a trend (ratio above trend_er) the
bot rides an adaptive moving average (KAMA), long while price is above a rising KAMA. In chop (ratio below
range_er) it switches to mean reversion: buy when 2-day RSI is deeply oversold, sell on a bounce or after
mr_max_hold days. Position size adapts to volatility, aiming at target_vol per position.
"""

from __future__ import annotations

import math

import numpy as np

from ..indicators import efficiency_ratio, kama, rolling_std, rsi
from .base import DailyStrategy, cap_gross, log_returns

FLAT, TREND, REVERSION = 0, 1, 2


class Adaptive(DailyStrategy):
    name, number, title = "adaptive", 14, "Adaptive (regime switching + volatility targeting)"
    defaults = {
        "er_len": 20,
        "trend_er": 0.30,
        "range_er": 0.15,
        "kama_len": 10,
        "rsi_len": 2,
        "rsi_buy": 10.0,
        "rsi_sell": 70.0,
        "mr_max_hold": 5,
        "vol_len": 20,
        "target_vol": 0.15,
        "slots": 5,
    }

    def warmup(self):
        return max(self.p["er_len"], self.p["kama_len"], self.p["vol_len"])

    def weights(self, panel):
        p = self.p
        c = panel.close
        T, N = c.shape
        er = efficiency_ratio(c, p["er_len"])  # NaN in the warm-up, which matches neither regime
        ma = kama(c, p["kama_len"])
        r = rsi(c, p["rsi_len"])
        with np.errstate(divide="ignore"):  # no movement at all: the cap below sizes it at 1
            size = np.clip(p["target_vol"] / (rolling_std(log_returns(c), p["vol_len"]) * math.sqrt(252)), 0, 1)
        mode, held = np.zeros(N, int), np.zeros(N, int)
        self.modes = np.zeros((T, N), int)
        for t in range(1, T):
            held = np.where(mode == REVERSION, held + 1, 0)
            mode = np.where((mode == TREND) & (c[t] < ma[t]), FLAT, mode)
            mode = np.where((mode == REVERSION) & ((r[t] > p["rsi_sell"]) | (held >= p["mr_max_hold"])), FLAT, mode)
            go_trend = (mode == FLAT) & (er[t] > p["trend_er"]) & (c[t] > ma[t]) & (ma[t] > ma[t - 1])
            go_revert = (mode == FLAT) & ~go_trend & (er[t] < p["range_er"]) & (r[t] < p["rsi_buy"])
            mode = np.where(go_trend, TREND, np.where(go_revert, REVERSION, mode))
            held = np.where(go_revert, 0, held)
            self.modes[t] = mode
        return cap_gross((self.modes != FLAT) * np.nan_to_num(size) / p["slots"])

    def describe(self, panel):
        mode = self.modes[-1]
        trend = [s for s, m in zip(panel.symbols, mode) if m == TREND]
        revert = [s for s, m in zip(panel.symbols, mode) if m == REVERSION]
        return [f"trend-following: {', '.join(trend) or 'none'}", f"mean-reverting: {', '.join(revert) or 'none'}"]
