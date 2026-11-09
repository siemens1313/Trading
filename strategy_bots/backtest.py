"""Daily backtest shared by every strategy in the daily bot.

Row t of a weight matrix is decided at the close of day t, traded at the open of day t+1 and held to
the open of day t+2, so no signal ever trades at a price it has already seen. Weights are fractions of
equity per symbol (negative = short). Costs are charged on every change of weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import Panel

TRADING_DAYS = 252


@dataclass
class Result:
    name: str
    dates: np.ndarray  # the day each position was held
    returns: np.ndarray  # net daily returns
    turnover: np.ndarray  # sum of |weight changes| at that day's open
    exposure: np.ndarray  # gross weight held

    def between(self, start: str | np.datetime64 | None = None, end: str | np.datetime64 | None = None) -> Result:
        """The days from start to end, both included (either may be None)."""
        keep = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            keep &= self.dates >= np.datetime64(start, "D")
        if end is not None:
            keep &= self.dates <= np.datetime64(end, "D")
        return Result(self.name, self.dates[keep], self.returns[keep], self.turnover[keep], self.exposure[keep])

    def metrics(self) -> dict[str, float]:
        return summarize(self.returns, self.turnover, self.exposure)


def run(panel: Panel, weights: np.ndarray, cost_bps: float = 5.0, name: str = "") -> Result:
    w = np.nan_to_num(np.asarray(weights, dtype=float))
    if w.shape != panel.close.shape:
        raise ValueError(f"weights shape {w.shape} != panel shape {panel.close.shape}")
    held = w[:-2]
    fwd = panel.open[2:] / panel.open[1:-1] - 1
    previous = np.vstack([np.zeros((1, w.shape[1])), held[:-1]])
    turnover = np.abs(held - previous).sum(axis=1)
    net = (held * fwd).sum(axis=1) - turnover * cost_bps / 1e4
    return Result(name, panel.dates[1:-1], net, turnover, np.abs(held).sum(axis=1))


def equal_weight(panel: Panel, name: str = "equal weight, all symbols") -> Result:
    """The benchmark: equal weight in every symbol, rebalanced daily at no cost, the way the strategies'
    weights are held. For a single symbol this is plain buy and hold."""
    weights = np.full(panel.close.shape, 1 / len(panel.symbols))
    return run(panel, weights, cost_bps=0.0, name=name)


def summarize(returns: np.ndarray, turnover: np.ndarray, exposure: np.ndarray) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return {k: math.nan for k in ("total_return", "cagr", "vol", "sharpe", "max_drawdown", "turnover", "exposure")}
    years = len(r) / TRADING_DAYS
    equity = np.cumprod(1 + r)
    sd = r.std(ddof=1)
    return {
        "total_return": float(equity[-1] - 1),
        "cagr": float(equity[-1] ** (1 / years) - 1) if equity[-1] > 0 else -1.0,
        "vol": float(sd * math.sqrt(TRADING_DAYS)),
        "sharpe": float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else math.nan,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "turnover": float(np.sum(turnover) / years),
        "exposure": float(np.mean(exposure)),
    }
