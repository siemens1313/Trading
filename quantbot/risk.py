"""Pre-trade checks and the drawdown kill switch."""
from __future__ import annotations

import math

from .config import RiskCfg
from .execution import Order


class RiskManager:
    def __init__(self, cfg: RiskCfg):
        self.cfg = cfg

    def check_orders(self, orders: list[Order], prices: dict[str, float],
                     last_closes: dict[str, float]) -> tuple[list[Order], list[tuple[Order, str]]]:
        ok, rejected = [], []
        for o in orders:
            # prices passed here are mids; ref is the last completed close
            px, ref = prices.get(o.symbol), last_closes.get(o.symbol)
            if px is None or not math.isfinite(px) or px <= 0:
                rejected.append((o, "invalid quote"))
            elif o.notional > self.cfg.max_order_notional:
                rejected.append((o, f"notional ${o.notional:,.0f} > max_order_notional"))
            elif ref and abs(px / ref - 1) > self.cfg.max_price_deviation:
                rejected.append((o, f"quote {px:.2f} deviates {px / ref - 1:+.1%} from last close {ref:.2f}"))
            else:
                ok.append(o)
        return ok, rejected

    def kill_switch(self, equity: float, peak: float) -> bool:
        return peak > 0 and (1 - equity / peak) >= self.cfg.kill_switch_drawdown
