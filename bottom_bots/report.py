"""Render results as Markdown or JSON."""

from __future__ import annotations

import json
from datetime import date
from typing import Iterable

from .bots import SENSOR_BOTS, AggregatorBot
from .bots.aggregator import Verdict
from .models import BotResult

AUTO_METRICS = "VIX, VIX3M, VXN and put/call ratios (Cboe); GEX and DIX (SqueezeMetrics); Fear & Greed (CNN); closes (Cboe, then Yahoo, then Stooq)"
MANUAL_ONLY = "si_pct_float, days_to_cover, borrow_fee, etf_flows_weekly, aaii_bearish, naaim, breadth_200dma, gamma_flip"
DISCLAIMER = "_A checklist for reading the market, not investment advice. The plan is a dry run; no order is ever sent._"


def render_markdown(verdict: Verdict, data_warnings: Iterable[str] = ()) -> str:
    v, p = verdict, verdict.plan
    lines = [f"# Bottom checklist — {v.as_of.isoformat()}", ""]
    unknown = f" · {v.unknown} unknown (up to {v.ceiling}/5 once that data is in)" if v.unknown else ""
    lines.append(f"**Score {v.score}/5**{unknown} · **{v.regime}** — {v.regime_quote}")
    if v.extremes:
        lines += ["", "At an extreme: " + "; ".join(v.extremes)]
    lines += ["", "| # | Bot | Status | Readings |", "|---|---|---|---|"]
    for r in v.results:
        readings = " · ".join(x.text(v.as_of) for x in r.readings)
        lines.append(f"| {r.item} | {r.title} | **{r.status.label}** | {readings} |")

    lines += ["", "## Plan — dry run", ""]
    held = p.persistent_score
    lines.append(f"- Persistent score: **{held}/5**")
    if p.runs_needed and v.score > held:
        lines.append(f"- Today's {v.score}/5 must hold for {p.runs_needed} more daily run(s) before it counts")
    lines.append(f"- Target: **{p.target:.0%}** of the reserve deployed (previous run: {p.previous_target:.0%})")
    lines.append(f"- New tranche: **+{p.increment:.0%}**" if p.increment > 0 else "- No new tranche")
    if p.gated:
        lines.append("- Held back by the money-back gate: item 5 (flows & sentiment) isn't met — “钱回来了才算数”")
    if p.paused:
        lines.append("- Conditions weakened since the last tranche: deployment paused (this bot never sells)")
    for order in p.orders:
        lines.append(f"  - proposed: {order['side']} {order['symbol']} ${order['notional']:,.2f}")
    if p.increment > 0 and not p.orders:
        lines.append("  - set plan.reserve_cash to turn the tranche into dollar amounts")

    if v.cautions:
        lines += ["", "## Cautions", ""] + [f"- {c}" for c in v.cautions]
    for r in v.results:
        lines += ["", *_section(r)]
    lines += _data_section(data_warnings)
    lines += ["", DISCLAIMER, ""]
    return "\n".join(lines)


def _data_section(data_warnings: Iterable[str]) -> list[str]:
    warnings = list(data_warnings)
    return ["", "## Data", ""] + [f"- {w}" for w in warnings] if warnings else []


def _section(r: BotResult) -> list[str]:
    lines = [f"## {r.item} · {r.title} — {r.status.label}", "", f"Rule: {r.rule}  ", f"Why: {r.summary}"]
    bullets = [f"- {f}" for f in r.flags]
    bullets += [f"- Extreme: {e}" for e in r.extremes]
    bullets += [f"- ⚠ {w}" for w in r.warnings]
    bullets += [f"- _{n}_" for n in r.notes]
    return lines + ([""] + bullets if bullets else [])


def render_sensors(results: list[BotResult], as_of: date, data_warnings: Iterable[str] = ()) -> str:
    lines = [f"# Sensor bots — {as_of.isoformat()}"]
    for r in results:
        lines += ["", *_section(r), "", "Readings: " + " · ".join(x.text(as_of) for x in r.readings)]
    lines += _data_section(data_warnings)
    return "\n".join(lines) + "\n"


def render_json(verdict: Verdict, data_warnings: Iterable[str] = ()) -> str:
    return json.dumps({**verdict.to_dict(), "data_warnings": list(data_warnings)}, indent=2, ensure_ascii=False)


def render_catalog() -> str:
    lines = ["Six bots: five sensors, one per checklist item, and a judge that scores them.", ""]
    for bot in SENSOR_BOTS:
        lines += [f"  {bot.item}. {bot.title}  [--bot {bot.key}]", f"     {bot.rule}"]
    lines += [f"  6. {AggregatorBot.title}", f"     {AggregatorBot.rule}", ""]
    lines += [f"Fetched automatically: {AUTO_METRICS}", f"Manual only (inputs/manual.toml): {MANUAL_ONLY}"]
    return "\n".join(lines)
