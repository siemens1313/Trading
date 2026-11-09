"""What every strategy in the daily bot provides, plus small shared helpers."""

from __future__ import annotations

import numpy as np

from ..data import Panel


class DailyStrategy:
    """Turns a panel into target weights, one row per day.

    Row t is decided at the close of day t and may use data up to day t only (see backtest.py); a test
    checks that for every strategy by cutting the future off and comparing.
    """

    name = ""
    number = 0  # row in the strategy table
    title = ""
    long_only = True
    defaults: dict = {}

    def __init__(self, **params):
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s): {', '.join(unknown)}")
        self.p = {**self.defaults, **params}

    def weights(self, panel: Panel) -> np.ndarray:
        raise NotImplementedError

    def warmup(self) -> int:
        """Days of history the strategy needs before it can trade; a fair comparison starts after it."""
        return 0

    def describe(self, panel: Panel) -> list[str]:
        """After weights(panel): lines on what the strategy sees on the last day."""
        return []


def first_fit(every: int, needed: int, window) -> int:
    """The first retraining day (a multiple of `every`) whose training window (lo, hi) spans `needed` rows."""
    for t in range(0, 1_000_000, every):
        lo, hi = window(t)
        if hi - lo >= needed:
            return t
    raise ValueError("the training window never reaches its minimum: raise train_days or lower min_train_days")


def slots(
    active: np.ndarray, n_slots: int, scores: np.ndarray | None = None, held: np.ndarray | None = None
) -> np.ndarray:
    """1/n_slots for each active name, keeping the best n_slots by score when more are active.

    Names in `held` that are still active keep their slots ahead of better-scored newcomers, so a noisy
    ranking doesn't churn the book every day.
    """
    w = np.zeros(active.shape)
    idx = np.flatnonzero(active)
    if len(idx) > n_slots:
        if scores is not None:
            idx = idx[np.argsort(-scores[idx], kind="stable")]
        if held is not None:
            keep = np.asarray(held, dtype=bool)[idx]
            idx = np.concatenate([idx[keep], idx[~keep]])
        idx = idx[:n_slots]
    w[idx] = 1 / n_slots
    return w


def cap_gross(w: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Scale rows whose gross exposure exceeds cap back down to it."""
    gross = np.abs(w).sum(axis=1, keepdims=True)
    return np.where(gross > cap, w * cap / np.where(gross > 0, gross, 1), w)


def log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(close), axis=0, prepend=np.nan)
