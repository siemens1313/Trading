"""Strategy 12: headline sentiment.

Each headline in the headlines file is scored with the finance lexicon in text.py and counted on the
first trading day on or after its date (known before the next open, when the bot trades). A per-symbol
exponential average with a `halflife`-day decay goes long above `entry` while the stock is above its
`trend_sma`-day average, and exits below `exit`. With no headlines file there is nothing to trade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..indicators import sma
from ..text import load_headlines, score
from .base import DailyStrategy, slots


class Sentiment(DailyStrategy):
    name, number, title = "sentiment", 12, "Sentiment analysis (headline lexicon)"
    defaults = {"headlines": "inputs/headlines.csv", "halflife": 3.0, "entry": 0.2, "exit": 0.0, "trend_sma": 50, "slots": 5}

    def warmup(self):
        return self.p["trend_sma"]

    def weights(self, panel):
        p = self.p
        T, N = panel.close.shape
        W = np.zeros((T, N))
        path = Path(p["headlines"])
        if not path.exists():
            self.note = f"{path} not found: no headlines, so no trades"
            return W
        total, count = np.zeros((T, N)), np.zeros((T, N))
        for day, symbol, text in load_headlines(path):
            if symbol not in panel.symbols:
                continue
            t = int(np.searchsorted(panel.dates, day))
            if t < T:
                total[t, panel.col(symbol)] += score(text)
                count[t, panel.col(symbol)] += 1
        news = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        decay = 0.5 ** (1 / p["halflife"])
        trend = panel.close > sma(panel.close, p["trend_sma"])
        level, holding = np.zeros(N), np.zeros(N, dtype=bool)
        self.levels = np.zeros((T, N))
        for t in range(T):
            level = decay * level + (1 - decay) * news[t]
            holding = np.where(holding, level >= p["exit"], (level > p["entry"]) & trend[t])
            self.levels[t] = level
            W[t] = slots(holding, p["slots"], level, held=W[t - 1] > 0 if t else None)
        return W

    def describe(self, panel):
        if getattr(self, "note", None):
            return [self.note]
        level = self.levels[-1]
        named = [(float(v), s) for s, v in zip(panel.symbols, level) if v]
        if not named:
            return ["no recent headlines for these symbols"]
        return ["headline tone (decayed): " + ", ".join(f"{s} {v:+.2f}" for v, s in sorted(named)[::-1])]
