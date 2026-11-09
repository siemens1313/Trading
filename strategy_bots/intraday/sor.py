"""Strategy 3: smart order routing.

The router splits an order across venues by effective price (quote plus taker fee, or minus a rebate)
and re-routes whatever didn't fill. Every miss costs something: while the order chases, the other venues
fade their quotes, so the shares that come back are bought higher. The adaptive router learns each
venue's real fill rate from its own fills (an exponential average) and ranks venues by price plus
(1 - fill rate) x the cost of a miss, so a venue that keeps showing size it doesn't honor drops down the
list. (A venue that has dropped off is never re-tested here; a real router probes it with small orders.)

A retail account can't route its own orders: Robinhood and most retail brokers route for you, mostly to
wholesale market makers. This is a simulator for studying routing, not something to run live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Venue:
    name: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    fee_bps: float = 0.0  # taker fee; negative for a rebate

    def effective(self, side: str) -> float:
        """What a share really costs (buy) or yields (sell) here, after the fee."""
        return self.ask * (1 + self.fee_bps / 1e4) if side == "buy" else self.bid * (1 - self.fee_bps / 1e4)


@dataclass
class Fill:
    venue: str
    shares: int
    price: float  # effective price, fee included


class SmartOrderRouter:
    def __init__(self, adaptive: bool = True, learning_rate: float = 0.2, miss_cost_bps: float = 5.0):
        """miss_cost_bps: what a share that doesn't fill costs, in bps -- the quotes it comes back to have
        moved, and it pays the spread again. A few bps for a liquid stock."""
        self.adaptive, self.learning_rate, self.miss_cost_bps = adaptive, learning_rate, miss_cost_bps
        self.fill_rate: dict[str, float] = {}

    def rank_price(self, v: Venue, side: str) -> float:
        """Effective price; for the adaptive router, worsened by the expected cost of not getting filled."""
        miss = (1 - self.fill_rate.get(v.name, 1.0)) * self.miss_cost_bps / 1e4 if self.adaptive else 0.0
        return v.effective(side) * (1 + miss if side == "buy" else 1 - miss)

    def plan(self, side: str, qty: int, venues: list[Venue]) -> list[tuple[Venue, int]]:
        """One routing pass: best ranked price first, each venue up to its displayed size."""
        ranked = sorted(venues, key=lambda v: self.rank_price(v, side), reverse=(side == "sell"))
        out, left = [], qty
        for v in ranked:
            take = min(left, v.ask_size if side == "buy" else v.bid_size)
            if take > 0:
                out.append((v, take))
                left -= take
            if left == 0:
                break
        return out

    def record(self, venue: str, sent: int, filled: int) -> None:
        if sent > 0:
            prior = self.fill_rate.get(venue, 1.0)
            self.fill_rate[venue] = prior + self.learning_rate * (filled / sent - prior)

    def execute(
        self,
        side: str,
        qty: int,
        quotes: Callable[[int], list[Venue]],
        fill: Callable[[Venue, str, int], int],
        passes: int = 10,
    ) -> list[Fill]:
        """Route, observe fills, learn, and re-route the rest. quotes(n) gives the venues on pass n."""
        fills: list[Fill] = []
        left = qty
        for n in range(passes):
            routed = self.plan(side, left, quotes(n))
            if not routed:
                break
            for v, sent in routed:
                got = min(sent, fill(v, side, sent))
                self.record(v.name, sent, got)
                if got:
                    fills.append(Fill(v.name, got, v.effective(side)))
                    left -= got
            if left == 0:
                break
        return fills


def simulate(
    rounds: int = 200, qty: int = 800, seed: int = 0, adaptive: bool = True, fade_bps: float = 1.0, passes: int = 10
) -> dict:
    """Buy orders against four venues. PHANTOM shows the best price but honors only 20% of its size, and
    every re-routing pass finds all quotes fade_bps worse, as liquidity providers notice the order.
    Cost is measured against the mid when the order arrived, so chasing shows up as cost."""
    rng = np.random.default_rng(seed)
    true_rate = {"NYSE": 0.95, "NASDAQ": 0.95, "IEX": 0.9, "PHANTOM": 0.2}
    router = SmartOrderRouter(adaptive=adaptive)
    costs, filled_total, flow = [], 0, {name: 0 for name in true_rate}
    mid = 100.0
    for _ in range(rounds):
        mid *= math.exp(rng.normal(0, 0.0005))

        def quotes(n: int, mid: float = mid) -> list[Venue]:
            m = mid * (1 + n * fade_bps / 1e4)
            return [
                Venue("NYSE", m - 0.01, m + 0.01, 300, 300, 0.3),
                Venue("NASDAQ", m - 0.01, m + 0.01, 300, 300, 0.3),
                Venue("IEX", m - 0.01, m + 0.02, 400, 400, 0.09),
                Venue("PHANTOM", m - 0.005, m + 0.005, 500, 500, 0.0),
            ]

        fills = router.execute(
            "buy", qty, quotes, lambda v, side, sent: int(rng.binomial(sent, true_rate[v.name])), passes
        )
        got = sum(f.shares for f in fills)
        filled_total += got
        for f in fills:
            flow[f.venue] += f.shares
        if got:
            avg = sum(f.shares * f.price for f in fills) / got
            costs.append((avg / mid - 1) * 1e4)
    return {
        "adaptive": adaptive,
        "fill_ratio": filled_total / (rounds * qty),
        "avg_cost_bps_vs_mid": float(np.mean(costs)),
        "flow_share": {k: v / max(filled_total, 1) for k, v in flow.items()},
        "learned_fill_rate": {k: round(v, 2) for k, v in router.fill_rate.items()},
    }
