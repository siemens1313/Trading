"""Transaction cost model: half-spread + slippage + square-root market impact."""
from __future__ import annotations

import math

from .config import CostCfg


def impact_bps(notional: float, adv_dollars: float, daily_vol: float, coef: float) -> float:
    if not (adv_dollars and adv_dollars > 0 and math.isfinite(daily_vol)):
        return 0.0
    return coef * daily_vol * math.sqrt(max(notional, 0.0) / adv_dollars) * 1e4


def fill_cost_bps(notional: float, adv_dollars: float, daily_vol: float, c: CostCfg, *, crossing_spread: bool = True) -> float:
    """Total adverse cost vs. mid (or vs. the quoted side if crossing_spread=False)."""
    spread = c.half_spread_bps if crossing_spread else 0.0
    return spread + c.slippage_bps + impact_bps(notional, adv_dollars, daily_vol, c.impact_coef)
