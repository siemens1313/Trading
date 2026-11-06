"""Alpha signals and their evidence-based combination.

Every alpha maps a price panel to a score panel (dates x symbols) using only data up to each date.
Scores are roughly unit-scale (z-scores clipped at +/-3).

The combiner never trusts an alpha it hasn't measured. For each alpha it tracks the realized
information coefficient (IC): the pooled correlation between the score on day t and the forward return
from the open of t+1 to the open of t+1+h. That return is only known after the open of t+1+h, so the IC
for day t is only used from then on. Alpha weights come from each alpha's rolling IC information
ratio, shrunk toward equal weights. The size of the forecast is set by the combined signal's own
realized IC, after a haircut, using Grinold's rule: E[r] = IC * sigma * score.
If the combined IC is <= 0, the forecast is zero and the book goes to cash.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AlphaCfg


def _xs_z(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date, clipped at +/-3."""
    mu, sd = df.mean(axis=1), df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-3, 3)


def _daily_vol(rets: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return rets.rolling(window, min_periods=window // 2).std()


def tsmom(C: pd.DataFrame, bench: str, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Own past return (skipping the most recent month) in units of its volatility. Not demeaned:
    when everything trends down, every score is negative, so the book de-risks."""
    r = C.shift(skip) / C.shift(lookback) - 1
    vol = _daily_vol(np.log(C).diff()) * math.sqrt(lookback - skip)
    return (r / vol).clip(-3, 3)


def xsmom(C: pd.DataFrame, bench: str, lookback: int = 126, skip: int = 5) -> pd.DataFrame:
    """Relative strength: risk-adjusted medium-term return ranked across the universe."""
    r = C.shift(skip) / C.shift(lookback) - 1
    vol = _daily_vol(np.log(C).diff())
    return _xs_z(r / vol)


def strev(C: pd.DataFrame, bench: str, lookback: int = 5, beta_window: int = 60) -> pd.DataFrame:
    """Short-term reversal of the market-hedged residual: fade names that moved too far vs. their beta."""
    ret = C.pct_change()
    m = ret[bench]
    mx, mm = ret.rolling(beta_window).mean(), m.rolling(beta_window).mean()
    cov = ret.mul(m, axis=0).rolling(beta_window).mean().sub(mx.mul(mm, axis=0))
    var = (m * m).rolling(beta_window).mean() - mm * mm
    beta = cov.div(var, axis=0)
    resid = ret - beta.mul(m, axis=0)
    move = resid.rolling(lookback).sum() / (_daily_vol(resid) * math.sqrt(lookback))
    move = move.replace([np.inf, -np.inf], np.nan)
    if bench in move:
        move[bench] = np.nan  # the benchmark has no residual by construction
    out = -_xs_z(move)
    if bench in out:
        out[bench] = 0.0
    return out


def lowvol(C: pd.DataFrame, bench: str, window: int = 63) -> pd.DataFrame:
    """Low-volatility anomaly: prefer calmer assets."""
    return -_xs_z(_daily_vol(np.log(C).diff(), window))


ALPHAS = {"tsmom": tsmom, "xsmom": xsmom, "strev": strev, "lowvol": lowvol}


def compute_alpha_scores(C: pd.DataFrame, cfg: AlphaCfg, bench: str) -> dict[str, pd.DataFrame]:
    return {k: ALPHAS[k](C, bench, **cfg.params.get(k, {})) for k in cfg.enabled}


def forward_returns(O: pd.DataFrame, h: int) -> pd.DataFrame:
    """At date t: open(t+1) -> open(t+1+h). Used ONLY for IC measurement, always lagged before use."""
    return O.shift(-(1 + h)) / O.shift(-1) - 1


def pooled_ic(score: pd.DataFrame, fwd: pd.DataFrame, window: int, min_obs: int) -> pd.Series:
    """Rolling pooled correlation between scores and forward returns, as of the day each observation is
    realized (i.e. shifted by 1 + h). Pooling over dates and symbols captures timing as well as ranking."""
    s, f = score.align(fwd, join="inner")
    mask = s.notna() & f.notna()
    s, f = s.where(mask), f.where(mask)
    daily = pd.DataFrame({"n": mask.sum(axis=1), "s": s.sum(axis=1), "f": f.sum(axis=1),
                          "ss": (s * s).sum(axis=1), "ff": (f * f).sum(axis=1), "sf": (s * f).sum(axis=1)})
    daily["obs"] = (daily["n"] > 2).astype(float)
    roll = daily.rolling(window, min_periods=1).sum()
    n = roll["n"].replace(0, np.nan)
    cov = roll["sf"] / n - (roll["s"] / n) * (roll["f"] / n)
    vs = roll["ss"] / n - (roll["s"] / n) ** 2
    vf = roll["ff"] / n - (roll["f"] / n) ** 2
    ic = cov / np.sqrt(vs * vf)
    return ic.where(roll["obs"] >= min_obs)


def daily_ic(score: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Cross-sectional (per-date) Pearson IC, for research reports."""
    s, f = score.align(fwd, join="inner")
    m = s.notna() & f.notna()
    s, f = s.where(m), f.where(m)
    sd, fd = s.sub(s.mean(axis=1), axis=0), f.sub(f.mean(axis=1), axis=0)
    return (sd * fd).sum(axis=1) / np.sqrt((sd * sd).sum(axis=1) * (fd * fd).sum(axis=1))


@dataclass
class Forecast:
    mu: pd.DataFrame          # alpha: expected relative return over the horizon (demeaned), date x symbol
    combined: pd.DataFrame    # combined cross-sectional z-score
    alpha_weights: pd.DataFrame
    alpha_ic: pd.DataFrame    # realized rolling cross-sectional IC per alpha, as known at each date
    combined_ic: pd.Series    # haircut IC actually used for sizing
    persistence: pd.Series    # autocorrelation of the combined score at lag h (drives cost amortization)
    scores: dict[str, pd.DataFrame]
    vol: pd.DataFrame         # daily vol used in mu


def _lag_autocorr(s: pd.DataFrame, lag: int, window: int) -> pd.Series:
    a, b = s, s.shift(lag)
    m = a.notna() & b.notna()
    a, b = a.where(m), b.where(m)
    d = pd.DataFrame({"n": m.sum(axis=1), "a": a.sum(axis=1), "b": b.sum(axis=1), "aa": (a * a).sum(axis=1),
                      "bb": (b * b).sum(axis=1), "ab": (a * b).sum(axis=1)}).rolling(window, min_periods=lag + 20).sum()
    n = d["n"].replace(0, np.nan)
    cov = d["ab"] / n - d["a"] / n * d["b"] / n
    return cov / np.sqrt((d["aa"] / n - (d["a"] / n) ** 2) * (d["bb"] / n - (d["b"] / n) ** 2))


def build_forecast(C: pd.DataFrame, O: pd.DataFrame, cfg: AlphaCfg, bench: str) -> Forecast:
    h = cfg.horizon
    scores = {k: _xs_z(v) for k, v in compute_alpha_scores(C, cfg, bench).items()}  # relative value only
    fwd = forward_returns(O, h)
    lag = 1 + h  # IC for day t is realized at the open of t+1+h -> usable from the close of t+1+h

    def rolling_ic(score):
        return daily_ic(score, fwd).rolling(cfg.ic_window, min_periods=cfg.ic_min_obs).mean().shift(lag)

    # Evidence: each alpha's realized cross-sectional IC. Negative IC -> zero weight (we never flip a signal).
    ic = pd.DataFrame({k: rolling_ic(s) for k, s in scores.items()})
    pos = ic.clip(lower=0).fillna(0.0)
    proven = (pos > 0).astype(float)
    data_w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0)
    equal_w = proven.div(proven.sum(axis=1).replace(0, np.nan), axis=0)  # prior only spreads over proven alphas
    w = ((1 - cfg.prior_weight) * data_w + cfg.prior_weight * equal_w).fillna(0.0)

    combined = _xs_z(sum(scores[k].fillna(0.0).mul(w[k], axis=0) for k in scores))
    comb_ic = (rolling_ic(combined).clip(lower=0) * cfg.ic_haircut).fillna(0.0)
    persistence = _lag_autocorr(combined, h, cfg.ic_window).clip(0, 0.99).fillna(0.0)

    vol = _daily_vol(np.log(C).diff())
    mu = combined.mul(comb_ic, axis=0) * vol * math.sqrt(h)
    return Forecast(mu=mu, combined=combined, alpha_weights=w, alpha_ic=ic, combined_ic=comb_ic,
                    persistence=persistence, scores=scores, vol=vol)
