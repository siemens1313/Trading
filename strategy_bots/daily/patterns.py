"""Strategy 11: chart-pattern recognition.

Swing pivots come from a zigzag on log prices, used only once confirmed. Bullish breakouts open a long:
inverse head and shoulders, double bottom, ascending triangle (all through their neckline or resistance)
and bull flag. Each sets a stop below the pattern and a measured-move target. The same detectors run on
the mirrored price find the bearish versions (head and shoulders, double top, descending triangle, bear
flag), which close the long.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from ..indicators import zigzag
from .base import DailyStrategy, cap_gross


def _crosses(s: np.ndarray, t: int, level: float) -> bool:
    return s[t - 1] <= level < s[t]


def detect(s: np.ndarray, pivots: list, t: int, p: dict) -> tuple[str, float, float] | None:
    """A bullish breakout on day t in the log-price series s: (pattern, stop, target), or None.

    `pivots` are (index, value, kind) confirmed by day t, oldest first.
    """
    tol, height = p["tolerance"], p["min_height"]
    kinds = tuple(k for _, _, k in pivots[-5:])
    vals = [v for _, v, _ in pivots[-5:]]
    if kinds == (-1, 1, -1, 1, -1):
        l1, h1, head, h2, l3 = vals
        neck = max(h1, h2)
        if head < min(l1, l3) - tol and abs(l1 - l3) <= 1.5 * tol and _crosses(s, t, neck):
            return "inverse head and shoulders", l3 - 0.01, neck + (neck - head)
    if kinds[-3:] == (-1, 1, -1):
        l1, h, l2 = vals[-3:]
        low = min(l1, l2)
        if abs(l2 - l1) <= tol and h - max(l1, l2) >= height and _crosses(s, t, h):
            return "double bottom", low - 0.01, h + (h - low)
    if kinds[-4:] == (1, -1, 1, -1):
        h1, l1, h2, l2 = vals[-4:]
        top = max(h1, h2)
        if abs(h2 - h1) <= tol / 1.5 and l2 > l1 + tol / 3 and _crosses(s, t, top):
            return "ascending triangle", l2 - 0.01, top + (top - l1)
    lo, hi = p["flag_min_days"], p["flag_max_days"]
    if t > hi + p["pole_days"]:
        window = s[t - hi : t]
        peak = t - hi + int(np.argmax(window))
        days = t - peak
        if lo <= days <= hi:
            top = s[peak]
            pole = top - s[peak - p["pole_days"] : peak].min()
            flag_low = s[peak:t].min()
            if pole >= p["pole_min"] and flag_low >= top - 0.5 * pole and _crosses(s, t, top):
                return "bull flag", flag_low - 0.01, s[t] + pole
    return None


class PatternRecognition(DailyStrategy):
    name, number, title = "patterns", 11, "Chart-pattern recognition (zigzag pivots)"
    defaults = {
        "swing": 0.05,
        "tolerance": 0.03,
        "min_height": 0.05,
        "pole_min": 0.10,
        "pole_days": 15,
        "flag_min_days": 3,
        "flag_max_days": 20,
        "max_hold": 40,
        "slots": 5,
    }

    def weights(self, panel):
        p = self.p
        y = np.log(panel.close)
        T, N = y.shape
        active = np.zeros((T, N), dtype=bool)
        self.signals: dict[str, str] = {}
        for j in range(N):
            s = y[:, j]
            ups = zigzag(s, p["swing"])
            downs = zigzag(-s, p["swing"])
            up_conf = [c for *_, c in ups]
            down_conf = [c for *_, c in downs]
            in_position, stop, target, held = False, 0.0, 0.0, 0
            for t in range(1, T):
                bull = detect(s, [x[:3] for x in ups[: bisect_right(up_conf, t)]], t, p)
                bear = detect(-s, [x[:3] for x in downs[: bisect_right(down_conf, t)]], t, p)
                if not in_position and bull:
                    (pattern, stop, target), in_position, held = bull, True, 0
                    if t == T - 1:
                        self.signals[panel.symbols[j]] = pattern
                elif in_position:
                    held += 1
                    if s[t] <= stop or s[t] >= target or held >= p["max_hold"] or bear:
                        in_position = False
                active[t, j] = in_position
        return cap_gross(active / p["slots"])

    def describe(self, panel):
        if not self.signals:
            return ["no new pattern completed on the last day"]
        return [f"new today: {s} {pattern}" for s, pattern in sorted(self.signals.items())]
