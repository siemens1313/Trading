"""Render results as Markdown or JSON."""

from __future__ import annotations

import html
import json
import re
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


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])_(?!\s)(.+?)(?<!\s)_(?![\w*])")  # _note_, not us_indices/daily_prices
_CODE = re.compile(r"`([^`]+)`")
_CELL = "padding:6px 8px;border:1px solid #ddd;text-align:left;vertical-align:top"


def _inline_html(text: str) -> str:
    text = _CODE.sub(r"<code>\1</code>", html.escape(text, quote=False))
    return _ITALIC.sub(r"<i>\1</i>", _BOLD.sub(r"<b>\1</b>", text))


def _inline_text(text: str) -> str:
    return _ITALIC.sub(r"\1", _BOLD.sub(r"\1", _CODE.sub(r"\1", text)))


def _table_row(line: str) -> list[str] | None:
    """Cells of a Markdown table row, or None for the |---| separator."""
    stripped = line.strip()
    if set(stripped) <= set("|-: "):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def markdown_to_html(markdown: str) -> str:
    """The report's Markdown (headings, one table, bullets, bold, italics) as email-safe HTML."""
    parts: list[str] = []
    items: list[tuple[str, bool]] = []
    rows: list[list[str]] = []

    def flush() -> None:
        if items:
            lis = "".join(f'<li style="margin-left:{18 if nested else 0}px">{text}</li>' for text, nested in items)
            parts.append(f'<ul style="margin:0 0 10px;padding-left:20px">{lis}</ul>')
            items.clear()
        if rows:
            head, *body = rows
            cells = "".join(f'<th style="{_CELL};background:#f2f2f2">{_inline_html(c)}</th>' for c in head)
            body_html = "".join(
                "<tr>" + "".join(f'<td style="{_CELL}">{_inline_html(c)}</td>' for c in row) + "</tr>" for row in body
            )
            parts.append(
                f'<table style="border-collapse:collapse;width:100%;font-size:13px;margin:0 0 10px">'
                f"<tr>{cells}</tr>{body_html}</table>"
            )
            rows.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if items:
                flush()
            row = _table_row(stripped)
            if row is not None:
                rows.append(row)
            continue
        if rows:
            flush()
        if stripped.startswith("- "):
            items.append((_inline_html(stripped[2:]), line.startswith("  ")))
            continue
        flush()
        if not stripped:
            continue
        if stripped.startswith("## "):
            parts.append(f'<h3 style="margin:16px 0 6px">{_inline_html(stripped[3:])}</h3>')
        elif stripped.startswith("# "):
            parts.append(f'<h2 style="margin:0 0 6px">{_inline_html(stripped[2:])}</h2>')
        else:
            parts.append(f'<p style="margin:0 0 8px">{_inline_html(stripped)}</p>')
    flush()
    style = "font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5;color:#1a1a1a"
    return f'<div style="{style};max-width:720px">\n' + "\n".join(parts) + "\n</div>\n"


def markdown_to_text(markdown: str) -> str:
    """The report's Markdown as plain text: no emphasis markers, table rows joined with ' · '."""
    lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            row = _table_row(stripped)
            if row is not None:
                lines.append(" · ".join(_inline_text(c) for c in row))
        elif stripped.startswith("#"):
            lines.append(_inline_text(stripped.lstrip("#").strip()))
        else:
            lines.append(_inline_text(line.rstrip()))
    return "\n".join(lines).strip() + "\n"


def render_catalog() -> str:
    lines = ["Six bots: five sensors, one per checklist item, and a judge that scores them.", ""]
    for bot in SENSOR_BOTS:
        lines += [f"  {bot.item}. {bot.title}  [--bot {bot.key}]", f"     {bot.rule}"]
    lines += [f"  6. {AggregatorBot.title}", f"     {AggregatorBot.rule}", ""]
    lines += [f"Fetched automatically: {AUTO_METRICS}", f"Manual only (inputs/manual.toml): {MANUAL_ONLY}"]
    return "\n".join(lines)
