"""Dual momentum + trend filter strategy.

At each rebalance:
  1. Momentum score = average total return over several lookbacks (e.g. 3/6/12 months).
  2. An asset is eligible only if its momentum is positive (absolute momentum) and its
     price is above its long moving average (trend filter).
  3. Hold the top-N eligible assets (relative momentum), weighted by inverse volatility.
  4. Scale the whole book so estimated portfolio volatility <= target_vol; cap each weight.
  Whatever is not allocated sits in cash. If nothing is eligible the bot is 100% cash.

compute_targets only ever sees the history passed to it, so there is no lookahead as long
as callers pass data up to and including the signal date.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import StrategyCfg


def required_history(p: StrategyCfg) -> int:
    return max(max(p.momentum_lookbacks) + p.skip_recent, p.trend_window, p.vol_window) + 1


def compute_targets(closes: pd.DataFrame, p: StrategyCfg) -> pd.Series | None:
    """Target portfolio weights (fraction of equity per symbol; remainder is cash).

    Returns None when there is not enough history to form a signal.
    """
    closes = closes.dropna(how="any")
    need = required_history(p)
    if len(closes) < need:
        return None
    hist = closes.iloc[-need:]
    last = hist.iloc[-1]
    ref = hist.iloc[-1 - p.skip_recent]

    mom = pd.concat([ref / hist.iloc[-1 - p.skip_recent - lb] - 1.0 for lb in p.momentum_lookbacks], axis=1).mean(axis=1)
    sma = hist.iloc[-p.trend_window:].mean()
    rets = np.log(hist.iloc[-(p.vol_window + 1):]).diff().dropna()
    vol = rets.std() * math.sqrt(252)

    eligible = (mom > 0) & (last > sma) & (vol > 0)
    weights = pd.Series(0.0, index=closes.columns)
    picks = mom[eligible].sort_values(ascending=False).index[: p.top_n]
    if len(picks) == 0:
        return weights

    raw = 1.0 / vol[picks]
    raw = raw / raw.sum()
    cov = rets[picks].cov().values * 252
    port_vol = float(math.sqrt(max(raw.values @ cov @ raw.values, 0.0)))
    scale = min(1.0, p.target_vol / port_vol) if port_vol > 0 else 1.0
    weights[picks] = (raw * scale).clip(upper=p.max_weight)
    return weights


def explain(closes: pd.DataFrame, p: StrategyCfg) -> pd.DataFrame:
    """Per-symbol signal table for logging / the status command."""
    closes = closes.dropna(how="any")
    need = required_history(p)
    if len(closes) < need:
        return pd.DataFrame()
    hist = closes.iloc[-need:]
    last, ref = hist.iloc[-1], hist.iloc[-1 - p.skip_recent]
    mom = pd.concat([ref / hist.iloc[-1 - p.skip_recent - lb] - 1.0 for lb in p.momentum_lookbacks], axis=1).mean(axis=1)
    sma = hist.iloc[-p.trend_window:].mean()
    vol = np.log(hist.iloc[-(p.vol_window + 1):]).diff().std() * math.sqrt(252)
    t = compute_targets(closes, p)
    return pd.DataFrame({
        "price": last, "momentum": mom, "above_sma": last > sma, "vol": vol, "target_w": t,
    }).sort_values("momentum", ascending=False)
