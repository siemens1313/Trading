"""Turns target weights into orders. Shared by the backtester and the live/paper engine so both behave the same."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import ExecutionCfg


@dataclass
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    ref_price: float   # price used for sizing (fill price may differ by slippage)

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


def _round_qty(q: float, fractional: bool) -> float:
    if q <= 0:
        return 0.0
    return math.floor(q * 1e6) / 1e6 if fractional else float(math.floor(q))


def plan_orders(targets: pd.Series, positions: dict[str, float], prices: dict[str, float],
                equity: float, ex: ExecutionCfg) -> list[Order]:
    """Orders that move `positions` to `targets` (weights of `equity`). Sells come before buys."""
    symbols = sorted(set(targets.index) | {s for s, q in positions.items() if q})
    orders: list[Order] = []
    for s in symbols:
        price = prices.get(s)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"no valid price for {s}")
        w = float(targets.get(s, 0.0))
        cur = float(positions.get(s, 0.0))
        tgt = _round_qty(w * equity / price, ex.fractional)
        delta = tgt - cur
        notional = abs(delta) * price
        closing = tgt == 0 and cur > 0
        if closing:
            if notional < 0.01:
                continue
        elif notional < max(ex.min_order_notional, ex.min_trade_weight * equity):
            continue
        orders.append(Order(s, "buy" if delta > 0 else "sell", abs(delta) if ex.fractional else float(round(abs(delta))), price))
    return [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]
