"""Target weights -> orders. Shared by the backtester and the live/paper engine."""
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
    ref_price: float

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


def round_qty(q: float, fractional: bool) -> float:
    """Round toward zero (never overshoot a target)."""
    a = abs(q)
    a = math.floor(a * 1e6) / 1e6 if fractional else float(math.floor(a))
    return math.copysign(a, q) if a else 0.0


def plan_orders(targets: pd.Series, positions: dict[str, float], prices: dict[str, float],
                equity: float, ex: ExecutionCfg) -> list[Order]:
    """Orders that move `positions` toward `targets` (signed weights of equity). Sells first."""
    symbols = sorted(set(targets.index[targets.fillna(0) != 0]) | {s for s, q in positions.items() if q})
    orders: list[Order] = []
    for s in symbols:
        price = prices.get(s)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"no valid price for {s}")
        cur = float(positions.get(s, 0.0))
        tgt = round_qty(float(targets.get(s, 0.0)) * equity / price, ex.fractional)
        delta = tgt - cur
        if abs(delta) < 1e-9:
            continue
        notional = abs(delta) * price
        closing = tgt == 0 and cur != 0
        if closing:
            if notional < 0.01:
                continue
        elif notional < max(ex.min_order_notional, ex.min_trade_weight * equity):
            continue
        qty = abs(delta) if ex.fractional else float(round(abs(delta)))
        orders.append(Order(s, "buy" if delta > 0 else "sell", qty, price))
    return [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]
