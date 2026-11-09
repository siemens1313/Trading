"""Strategy 9: volatility breakout.

When a stock's Bollinger bandwidth has recently sat in the bottom `squeeze_pct` percent of its last
`squeeze_lookback` days (a volatility squeeze), a close above the upper band opens a long. It exits on a
trailing stop of `atr_stop` ATRs or after max_hold days. Positions are sized to target_vol. The table's
options and futures versions trade volatility itself; this stock-only bot trades the breakout it predicts.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..indicators import atr, bollinger, rolling_std
from .base import DailyStrategy, cap_gross, log_returns


class VolatilityBreakout(DailyStrategy):
    name, number, title = "vol_breakout", 9, "Volatility breakout (squeeze, then breakout)"
    defaults = {
        "bb_len": 20,
        "bb_k": 2.0,
        "squeeze_lookback": 120,
        "squeeze_pct": 20.0,
        "squeeze_window": 10,
        "atr_len": 14,
        "atr_stop": 3.0,
        "max_hold": 40,
        "target_vol": 0.20,
        "slots": 5,
    }

    def warmup(self):
        return self.p["bb_len"] + self.p["squeeze_lookback"]

    def weights(self, panel):
        p = self.p
        c = panel.close
        T, N = c.shape
        _, upper, _, width = bollinger(c, p["bb_len"], p["bb_k"])
        L = p["squeeze_lookback"]
        floor = np.full(c.shape, np.nan)
        if T >= L:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # early windows are all-NaN warm-up
                floor[L - 1 :] = np.nanpercentile(sliding_window_view(width, L, axis=0), p["squeeze_pct"], axis=-1)
        squeeze = np.cumsum(width <= floor, axis=0)
        k = p["squeeze_window"]
        recent = np.zeros(c.shape, dtype=bool)  # a squeeze in the k days before today
        recent[1:] = squeeze[:-1] - np.vstack([np.zeros((k, N)), squeeze[:-1]])[: T - 1] > 0
        a = atr(panel.high, panel.low, c, p["atr_len"])
        with np.errstate(divide="ignore"):  # no movement at all: the cap below sizes it at 1
            size = np.clip(p["target_vol"] / (rolling_std(log_returns(c), 20) * math.sqrt(252)), 0, 1)
        active = np.zeros(c.shape, dtype=bool)
        for j in range(N):
            in_position, stop, held = False, 0.0, 0
            for t in range(T):
                if not in_position:
                    if recent[t, j] and c[t, j] > upper[t, j] and np.isfinite(a[t, j]):
                        in_position, stop, held = True, c[t, j] - p["atr_stop"] * a[t, j], 0
                else:
                    held += 1
                    if c[t, j] < stop or held >= p["max_hold"]:
                        in_position = False
                    else:
                        stop = max(stop, c[t, j] - p["atr_stop"] * a[t, j])
                active[t, j] = in_position
        return cap_gross(active * np.nan_to_num(size) / p["slots"])
