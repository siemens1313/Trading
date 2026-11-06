"""Glue: forecast + risk model + optimizer -> target weights for one decision date."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alphas import Forecast
from .config import Config
from .optimizer import solve
from .riskmodel import covariance


@dataclass
class Decision:
    weights: pd.Series
    mu: pd.Series
    ex_ante_vol: float
    scaled_by: float
    combined_ic: float
    alpha_weights: pd.Series


def rolling_beta(rets: pd.DataFrame, bench: str, window: int) -> pd.DataFrame:
    m = rets[bench]
    mx, mm = rets.rolling(window, min_periods=window // 2).mean(), m.rolling(window, min_periods=window // 2).mean()
    cov = rets.mul(m, axis=0).rolling(window, min_periods=window // 2).mean() - mx.mul(mm, axis=0)
    var = (m * m).rolling(window, min_periods=window // 2).mean() - mm * mm
    return cov.div(var, axis=0)


def decide(T: pd.Timestamp, fc: Forecast, rets: pd.DataFrame, ewma_vol: pd.DataFrame,
           w0: pd.Series, cfg: Config, beta: pd.DataFrame | None = None) -> Decision | None:
    """Target weights at the close of T. Returns None if the risk model can't be built yet.

    Expected return = prior (long-only: beta * premium) + alpha (evidence-weighted relative value)."""
    syms = list(rets.columns)
    p, c, h = cfg.portfolio, cfg.costs, cfg.alphas.horizon
    alpha = fc.mu.loc[T].reindex(syms).fillna(0.0)
    if p.long_only:
        b = (beta.loc[T] if beta is not None else pd.Series(1.0, index=syms)).reindex(syms)
        mu = b * p.prior_premium / 252 * h + alpha
    else:
        mu = alpha
    vol = ewma_vol.loc[T].reindex(syms)
    ok = mu.notna() & vol.notna() & (vol > 0)
    w = pd.Series(0.0, index=syms)
    ic = float(fc.combined_ic.loc[T])
    aw = fc.alpha_weights.loc[T]
    if ok.sum() == 0 or (not p.long_only and ic <= 0):
        return Decision(w, mu.fillna(0.0), 0.0, 1.0, ic, aw)
    live = [s for s in syms if ok[s]]
    try:
        cov = covariance(rets.loc[:T, live], vol[live], cfg.risk_model.corr_window)
    except ValueError:
        return None
    n = len(live)
    hi = np.full(n, p.max_weight)
    lo = np.zeros(n) if p.long_only else -hi
    kappa = p.tcost_aversion * max(1.0 - float(fc.persistence.loc[T]), 0.02)  # amortize over signal life
    res = solve(mu[live].values, cov, w0.reindex(live).fillna(0.0).values, np.full(n, c.linear_bps / 1e4),
                horizon=h, risk_aversion=p.risk_aversion, tcost_aversion=kappa,
                lo=lo, hi=hi, max_gross=p.max_gross, target_vol=p.target_vol, calibrate=p.calibrate_risk)
    w[live] = res.w
    w[w.abs() < 1e-6] = 0.0
    return Decision(w, mu.fillna(0.0), res.ex_ante_vol, res.scaled_by, ic, aw)
