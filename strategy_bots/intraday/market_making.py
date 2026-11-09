"""Strategy 4: market making (Avellaneda-Stoikov).

Quotes a bid and an ask around a reservation price that leans away from inventory: holding stock moves
both quotes down so the next fill is more likely a sale. The spread widens with volatility and with the
time left in the session; inventory is capped and flattened at the close. The statistical-arbitrage
term from the table is optional: with a correlated reference stock, the fair value shifts by beta times the
reference's move over the last few bars that this stock hasn't followed yet, capped at the half-spread so
the skew leans the quotes without crossing the mid. (Uncapped, a 15-minute SPY move was often 20 bps
while the half-spread was a few, so the "quotes" crossed the market and paid for it.)

A quote is filled only when the next bar trades through it. That still flatters the strategy: a real
market maker joins a queue, gets picked off by faster traders, and earns rebates a retail account can't.
Simulation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..data import Session


@dataclass
class MarketMakingResult:
    pnl: float
    fills: int
    max_inventory: int
    spread_captured: float  # $ from round trips at quoted prices, before the closing flatten
    inventory: list[int] = field(default_factory=list)


class MarketMaker:
    defaults = {
        "gamma": 0.001,  # risk aversion, per bp^2
        "k": 0.25,  # order-arrival intensity, per bp: higher means tighter quotes
        "lot": 10,
        "max_lots": 5,
        "vol_bars": 20,
        "min_half_spread_bps": 2.0,
        "ref_beta": 0.0,  # weight of the reference stock's unfollowed move; 0 disables stat arb
        "ref_bars": 3,
    }

    def __init__(self, **params):
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ValueError(f"market_making: unknown parameter(s): {', '.join(unknown)}")
        self.p = {**self.defaults, **params}

    def quotes(self, mid: float, sigma_bps: float, lots: int, bars_left: int, shift_bps: float = 0.0) -> tuple[float, float]:
        """Bid and ask. Everything is in basis points of mid: sigma is per bar, the shift is the stat-arb term."""
        p = self.p
        risk = p["gamma"] * sigma_bps**2 * bars_left
        half = max(risk / 2 + math.log(1 + p["gamma"] / p["k"]) / p["gamma"], p["min_half_spread_bps"])
        shift = min(max(shift_bps, -half), half)  # lean on the signal, but never quote through the mid for it
        reservation = shift - lots * risk
        return mid * (1 + (reservation - half) / 1e4), mid * (1 + (reservation + half) / 1e4)

    def run(self, s: Session, reference: Session | None = None) -> MarketMakingResult:
        p = self.p
        n = len(s)
        lots, cash, fills, max_lots, captured = 0, 0.0, 0, 0, 0.0
        path = []
        logc = np.log(s.close)
        ref = np.log(reference.close) if reference is not None and len(reference) == n else None
        buys, sells = [], []
        for k in range(p["vol_bars"], n - 1):
            sigma = np.std(np.diff(logc[k - p["vol_bars"] : k + 1])) * 1e4
            shift = 0.0
            if ref is not None and p["ref_beta"]:
                m = p["ref_bars"]
                shift = (p["ref_beta"] * (ref[k] - ref[k - m]) - (logc[k] - logc[k - m])) * 1e4
            bid, ask = self.quotes(s.close[k], sigma, lots, n - 1 - k, shift)
            if s.low[k + 1] < bid and lots < p["max_lots"]:
                lots, cash, fills = lots + 1, cash - bid * p["lot"], fills + 1
                buys.append(bid)
            if s.high[k + 1] > ask and lots > -p["max_lots"]:
                lots, cash, fills = lots - 1, cash + ask * p["lot"], fills + 1
                sells.append(ask)
            max_lots = max(max_lots, abs(lots))
            path.append(lots)
        pairs = min(len(buys), len(sells))
        captured = (sum(sells[:pairs]) - sum(buys[:pairs])) * p["lot"]
        last = s.close[-1]
        flatten = last * (1 - math.copysign(p["min_half_spread_bps"] / 1e4, lots)) if lots else last
        cash += lots * p["lot"] * flatten
        return MarketMakingResult(cash, fills, max_lots, captured, path)
