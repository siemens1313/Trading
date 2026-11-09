"""Bot 2 — Cboe put/call: is retail fear at the late-selloff extreme? (article §2)"""

from __future__ import annotations

from ..logic import status_of
from ..models import BotResult
from ..series import percentile_rank
from .base import SensorBot


class PutCallBot(SensorBot):
    key = "put_call"
    item = 2
    title = "Put/Call bot"
    rule = "Cboe Equity put/call ≥ 0.80"

    def evaluate(self) -> BotResult:
        c = self.cfg["put_call"]
        equity = self.fresh("pc_equity")
        total = self.fresh("pc_total")
        index = self.fresh("pc_index")
        verdict = None if equity is None else equity.value >= c["equity_capitulation"]

        flags: list[str] = []
        if equity is None:
            summary = "No fresh Equity put/call reading (Cboe Daily Market Statistics)"
        else:
            relation = "≥" if verdict else "<"
            summary = f"Equity P/C {equity.value:.2f} {relation} {c['equity_capitulation']:.2f}"
            if equity.value < c["equity_euphoria"]:
                flags.append(f"Equity P/C below {c['equity_euphoria']:.2f}: call-chasing is overheated")
            history = [p.value for p in self.store.series("pc_equity") if p.day < equity.day]
            history = history[-c["percentile_sessions"]:]
            if len(history) >= c["percentile_min_points"]:
                rank = percentile_rank(history, equity.value)
                flags.append(f"Equity P/C is above {rank:.0%} of the previous {len(history)} sessions")

        notes = [
            "Retail sentiment lives in the Equity ratio; an Index ratio near 1 is normal institutional hedging",
            "0.8–1.0 is a reference zone, not a buy/sell signal — read it with options flow, OI changes, IV, "
            "price location and volume (article §2)",
        ]
        readings = [
            self.reading("Equity", equity),
            self.reading("Total", total),
            self.reading("Index", index),
        ]
        return self.result(status_of(verdict), summary, readings, flags, notes)
