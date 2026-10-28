"""Manual inputs: the numbers you read off the article's dashboards, one dated row each.

    [[aaii_bearish]]
    date = 2026-10-22
    value = 45.8
    source = "AAII"        # optional, shown in the report

Prices go under a symbol: [[close.SOXX]].
"""

from __future__ import annotations

import tomllib
from datetime import date, datetime
from pathlib import Path

from .store import MANUAL, SeriesStore

METRICS = {
    "vix": "VIX close",
    "vix3m": "VIX3M close",
    "vxn": "VXN close",
    "pc_equity": "Cboe equity put/call ratio",
    "pc_total": "Cboe total put/call ratio",
    "pc_index": "Cboe index put/call ratio",
    "gex": "net dealer gamma, $bn per 1% move (negative = short gamma)",
    "dix": "SqueezeMetrics DIX, %",
    "gamma_flip": "gamma flip level, in points of [gamma].underlying",
    "si_pct_float": "short interest, % of float",
    "days_to_cover": "short interest / average daily volume",
    "borrow_fee": "borrow fee, % annualised",
    "etf_flows_weekly": "combined weekly net flow of [flows_sentiment].etfs, $mn (outflow < 0)",
    "aaii_bearish": "AAII sentiment survey, bearish %",
    "fear_greed": "CNN Fear & Greed index, 0-100",
    "naaim": "NAAIM exposure index",
    "breadth_200dma": "% of S&P 500 members above their 200-day average",
}

FORBIDDEN = {
    "short_sale_volume": (
        "Short Sale Volume ≠ Short Interest (article §4): daily short-sale volume is not a "
        "net short position. Enter si_pct_float / days_to_cover instead."
    ),
}


class ManualInputError(ValueError):
    pass


def load_into(store: SeriesStore, path: Path) -> int:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    count = 0
    for key, rows in data.items():
        if key in FORBIDDEN:
            raise ManualInputError(f"{path}: {FORBIDDEN[key]}")
        if key == "close":
            if not isinstance(rows, dict):
                raise ManualInputError(f"{path}: prices go under a symbol, e.g. [[close.SOXX]]")
            for symbol, symbol_rows in rows.items():
                count += _add_rows(store, f"close:{symbol.upper()}", symbol_rows, path)
            continue
        if key not in METRICS:
            known = ", ".join(sorted(METRICS))
            raise ManualInputError(f"{path}: unknown metric '{key}' (known: {known}, close.<SYMBOL>)")
        count += _add_rows(store, key, rows, path)
    return count


def _add_rows(store: SeriesStore, metric: str, rows: object, path: Path) -> int:
    if not isinstance(rows, list):
        raise ManualInputError(f"{path}: '{metric}' must be an array of tables — write [[{metric}]]")
    for i, row in enumerate(rows, 1):
        day, value = row.get("date"), row.get("value")
        if isinstance(day, datetime):
            day = day.date()
        if not isinstance(day, date):
            raise ManualInputError(f"{path}: {metric} row {i} needs date = YYYY-MM-DD (unquoted)")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManualInputError(f"{path}: {metric} row {i} needs a numeric value")
        label = row.get("source")
        source = f"{MANUAL}:{label}" if isinstance(label, str) and label else MANUAL
        store.add(metric, day, float(value), source)
    return len(rows)
