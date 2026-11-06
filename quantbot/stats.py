"""Performance statistics with an honest view of uncertainty."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

N = NormalDist()
EULER = 0.5772156649


def sharpe(r: pd.Series, periods: int = 252) -> float:
    sd = r.std()
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given sample length, skew and kurtosis (Bailey & Lopez de Prado).
    Sharpes here are per-period (daily), converted internally."""
    r = r.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    sk, ku = float(r.skew()), float(r.kurt()) + 3.0
    denom = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    return N.cdf((sr - sr_benchmark / math.sqrt(252)) * math.sqrt(n - 1) / denom)


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpe_std: float) -> float:
    """PSR against the Sharpe you'd expect from the best of `n_trials` skill-less strategies.
    trial_sharpe_std: cross-trial std of annualized Sharpes."""
    if n_trials <= 1:
        return probabilistic_sharpe(r, 0.0)
    e_max = trial_sharpe_std * ((1 - EULER) * N.inv_cdf(1 - 1 / n_trials)
                                + EULER * N.inv_cdf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, e_max)


def bootstrap_sharpe_ci(r: pd.Series, n_boot: int = 2000, block: int = 21, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI for annualized Sharpe (keeps volatility clustering)."""
    x = r.dropna().values
    n = len(x)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    s = x[idx]
    sr = s.mean(axis=1) / s.std(axis=1, ddof=1) * math.sqrt(252)
    return float(np.quantile(sr, alpha / 2)), float(np.quantile(sr, 1 - alpha / 2))


def summarize(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = float(r.std() * math.sqrt(252))
    dd = equity / equity.cummax() - 1
    down = r[r < 0].std() * math.sqrt(252)
    lo, hi = bootstrap_sharpe_ci(r)
    return {
        "start": equity.index[0].date(), "end": equity.index[-1].date(),
        "total_return": float(total), "cagr": float(cagr), "ann_vol": vol, "sharpe": sharpe(r),
        "sharpe_ci": (lo, hi), "sortino": float(r.mean() * 252 / down) if down > 0 else float("nan"),
        "max_drawdown": float(dd.min()), "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
        "psr": probabilistic_sharpe(r), "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()), "worst_day": float(r.min()),
    }
