"""Bot 6 — the judge: counts the five checklist items and turns a persistent score into a staged plan.

The plan is a dry run. It says how much of the cash reserve the checklist would have
deployed by now; it never sends an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..history import RunRecord
from ..models import BotResult, Status

REGIMES = {
    "capitulation": ("CAPITULATION-LIKE", "四五条同时出现，才接近历史上那种投降式底部"),
    "building": ("BUILDING", "the article doesn't define three; treated as not yet a capitulation"),
    "oversold": ("OVERSOLD ONLY", "满足一两条，只能算超跌"),
    "none": ("NO SIGNAL", "no checklist item met"),
}


@dataclass
class Plan:
    persistent_score: int
    runs_needed: int  # further runs today's score must hold before it counts
    target: float  # cumulative share of the reserve deployed
    previous_target: float
    increment: float
    gated: bool  # money-back gate held the target down
    paused: bool  # score fell below the level that set the current target
    reserve_cash: float
    orders: list[dict] = field(default_factory=list)


@dataclass
class Verdict:
    as_of: date
    score: int
    unknown: int
    regime: str
    regime_quote: str
    results: list[BotResult]
    plan: Plan
    cautions: list[str]
    extremes: list[str]

    @property
    def ceiling(self) -> int:
        return self.score + self.unknown

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "score": self.score,
            "unknown": self.unknown,
            "ceiling": self.ceiling,
            "regime": self.regime,
            "regime_quote": self.regime_quote,
            "extremes": self.extremes,
            "cautions": self.cautions,
            "plan": self.plan.__dict__,
            "results": [r.to_dict() for r in self.results],
        }


class AggregatorBot:
    key = "aggregator"
    item = 6
    title = "Capitulation judge"
    rule = "1–2 items met = oversold only; 4–5 met together = close to a capitulation bottom"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def evaluate(self, results: list[BotResult], previous: list[RunRecord], as_of: date) -> Verdict:
        score = sum(r.status is Status.MET for r in results)
        unknown = sum(r.status is Status.UNKNOWN for r in results)
        regime, quote = REGIMES[self.regime(score)]
        money_back = any(r.key == "flows_sentiment" and r.status is Status.MET for r in results)
        return Verdict(
            as_of=as_of,
            score=score,
            unknown=unknown,
            regime=regime,
            regime_quote=quote,
            results=results,
            plan=build_plan(score, money_back, previous, self.cfg),
            cautions=[f"{r.title}: {w}" for r in results for w in r.warnings],
            extremes=[e for r in results for e in r.extremes],
        )

    def regime(self, score: int) -> str:
        agg = self.cfg["aggregator"]
        if score >= agg["capitulation_min"]:
            return "capitulation"
        if score >= 3:
            return "building"
        if score >= agg["oversold_min"]:
            return "oversold"
        return "none"

    @staticmethod
    def record(verdict: Verdict) -> RunRecord:
        return RunRecord(
            as_of=verdict.as_of,
            score=verdict.score,
            unknown=verdict.unknown,
            persistent=verdict.plan.persistent_score,
            target=verdict.plan.target,
            increment=verdict.plan.increment,
            statuses={r.key: r.status.value for r in verdict.results},
        )


def build_plan(score: int, money_back: bool, previous: list[RunRecord], cfg: dict) -> Plan:
    agg, plan = cfg["aggregator"], cfg["plan"]
    n = agg["persistence_runs"]
    recent = [r.score for r in previous[-(n - 1):]] if n > 1 else []
    # Runs that never happened count as score 0: a first run can't deploy anything.
    padded = [0] * (n - 1 - len(recent)) + recent
    persistent = min(padded + [score])

    streak = 1
    for past in reversed(recent):
        if past < score:
            break
        streak += 1
    runs_needed = max(0, n - streak)

    tranches = plan["tranches"]
    target_now = tranches[persistent]
    gate_level = agg["capitulation_min"] - 1
    gated = False
    if agg["money_back_gate"] and not money_back and target_now > tranches[gate_level]:
        target_now, gated = tranches[gate_level], True

    previous_target = previous[-1].target if previous else 0.0
    target = max(previous_target, target_now)  # tranches only ratchet up; this bot never sells
    increment = round(target - previous_target, 6)
    paused = score < (previous[-1].persistent if previous else 0)

    reserve = float(plan["reserve_cash"])
    orders = []
    if increment > 0 and reserve > 0:
        orders = [
            {"symbol": symbol, "side": "BUY", "notional": round(increment * reserve * weight, 2)}
            for symbol, weight in plan["instruments"].items()
        ]
    return Plan(
        persistent_score=persistent,
        runs_needed=runs_needed,
        target=target,
        previous_target=previous_target,
        increment=increment,
        gated=gated,
        paused=paused,
        reserve_cash=reserve,
        orders=orders,
    )
