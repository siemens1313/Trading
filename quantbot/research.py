"""Alpha research: is there an edge, how big, how stable, how fast does it decay, and does it survive costs?"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from .alphas import compute_alpha_scores, daily_ic, forward_returns, _xs_z
from .backtest import run_backtest
from .config import Config
from .data import align
from .stats import deflated_sharpe


def alpha_report(bars, cfg: Config, horizons=(1, 5, 21), standalone: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    C, O = align(bars, "close"), align(bars, "open")
    scores = {k: _xs_z(v) for k, v in compute_alpha_scores(C, cfg.alphas, cfg.universe.benchmark).items()}
    rows = {}
    for k, s in scores.items():
        r = {}
        for h in horizons:
            d = daily_ic(s, forward_returns(O, h)).dropna()
            r[f"IC {h}d"] = d.mean()
            if h == cfg.alphas.horizon:
                r["ICIR"] = d.mean() / d.std()
                r["t-stat"] = d.mean() / d.std() * math.sqrt(len(d) / h)  # overlap-adjusted
                r["hit rate"] = (d > 0).mean()
        a, b = s.stack(), s.shift(cfg.alphas.horizon).stack()
        j = pd.concat([a, b], axis=1).dropna()
        r[f"persist {cfg.alphas.horizon}d"] = j.corr().iloc[0, 1]
        if standalone:
            c = copy.deepcopy(cfg)
            c.alphas.enabled = [k]
            try:
                res = run_backtest(bars, c)
                r["solo Sharpe"] = res.metrics["sharpe"]
                r["solo turnover"] = res.metrics["turnover_per_year"]
            except ValueError:
                r["solo Sharpe"] = np.nan
        rows[k] = r
    corr = pd.DataFrame({k: s.stack() for k, s in scores.items()}).corr()
    return pd.DataFrame(rows).T, corr


def deflated(res, n_trials: int, trial_sharpe_std: float) -> float:
    return deflated_sharpe(res.equity.pct_change().dropna(), n_trials, trial_sharpe_std)


def fmt_report(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.startswith("IC") or c == "ICIR":
            out[c] = out[c].map(lambda x: f"{x:+.3f}")
        elif c in ("hit rate",):
            out[c] = out[c].map(lambda x: f"{x:.0%}")
        elif c.startswith("persist") or c in ("t-stat", "solo Sharpe"):
            out[c] = out[c].map(lambda x: f"{x:+.2f}")
        elif c == "solo turnover":
            out[c] = out[c].map(lambda x: f"{x:.1f}x")
    return out


def overlay_attribution(bars, cfg: Config, full=None) -> dict:
    """Does the alpha overlay add value net of costs? Compare the full book with the same book run on
    the beta prior alone (alpha forecasts zeroed). Only meaningful in long-only mode (L/S has no prior)."""
    base_cfg = copy.deepcopy(cfg)
    base_cfg.alphas.ic_haircut = 0.0
    full = full or run_backtest(bars, cfg)
    base = run_backtest(bars, base_cfg)
    act = (full.equity.pct_change() - base.equity.pct_change()).dropna()
    te = act.std() * math.sqrt(252)
    return {"baseline_sharpe": base.metrics["sharpe"], "full_sharpe": full.metrics["sharpe"],
            "baseline_turnover": base.metrics["turnover_per_year"], "full_turnover": full.metrics["turnover_per_year"],
            "active_return": act.mean() * 252, "tracking_error": te,
            "ir": act.mean() * 252 / te if te > 0 else float("nan"),
            "t_stat": act.mean() / act.std() * math.sqrt(len(act)) if act.std() > 0 else float("nan")}
