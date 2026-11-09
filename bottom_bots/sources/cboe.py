"""Cboe: VIX / VIX3M / VXN daily history, and the daily put/call ratios."""

from __future__ import annotations

import csv
import io
import json
import re
import time
from datetime import date, timedelta
from typing import Callable

from .http import FetchError
from .util import parse_day

INDEX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
DAILY_OPTIONS_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{day}_daily_options"
VOL_INDICES = {"vix": "VIX", "vix3m": "VIX3M", "vxn": "VXN"}

_RATIO = re.compile(r"^(TOTAL|INDEX|EQUITY) PUT ?/ ?CALL RATIO$")

Fetch = Callable[[str], bytes]


def parse_index_history(text: str, symbol: str) -> list[tuple[date, float]]:
    header: list[str] | None = None
    day_col = value_col = 0
    out = []
    for row in csv.reader(io.StringIO(text)):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if header is None:
            upper = [c.upper() for c in cells]
            if "DATE" in upper:
                header = upper
                day_col = upper.index("DATE")
                if "CLOSE" in upper:
                    value_col = upper.index("CLOSE")
                elif symbol.upper() in upper:
                    value_col = upper.index(symbol.upper())
                else:
                    value_col = len(upper) - 1
            continue
        try:
            out.append((parse_day(cells[day_col]), float(cells[value_col])))
        except (ValueError, IndexError):
            continue
    if header is None:
        raise ValueError(f"no DATE column in the {symbol} history CSV")
    return out


def fetch_vol_index(metric: str, fetch: Fetch) -> list[tuple[str, date, float, str]]:
    symbol = VOL_INDICES[metric]
    text = fetch(INDEX_HISTORY_URL.format(symbol=symbol)).decode("utf-8-sig")
    return [(metric, day, value, "cboe") for day, value in parse_index_history(text, symbol)]


def parse_put_call(payload: object) -> dict[str, float]:
    """Find the TOTAL / INDEX / EQUITY ratios anywhere in the daily statistics JSON."""
    found: dict[str, float] = {}
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            name = node.get("name")
            if isinstance(name, str) and "value" in node:
                match = _RATIO.match(" ".join(name.upper().split()))
                if match:
                    try:
                        found.setdefault(f"pc_{match.group(1).lower()}", float(node["value"]))
                    except (TypeError, ValueError):
                        pass
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _ratios_for(day: date, fetch: Fetch) -> dict[str, float]:
    return parse_put_call(json.loads(fetch(DAILY_OPTIONS_URL.format(day=day.isoformat()))))


def fetch_put_call(as_of: date, fetch: Fetch, search_days: int = 7) -> list[tuple[str, date, float, str]]:
    """Ratios for the latest published session on or before `as_of`."""
    last_error = ""
    for back in range(search_days):
        day = as_of - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        try:
            ratios = _ratios_for(day, fetch)
        except FetchError as exc:
            if exc.status is None:
                raise  # host unreachable: earlier days won't load either
            last_error = f" (last error: {exc})"
            continue
        except ValueError as exc:
            last_error = f" (last error: {exc})"
            continue
        if ratios:
            return [(metric, day, value, "cboe") for metric, value in ratios.items()]
    raise FetchError(f"no put/call ratios published in the {search_days} days to {as_of}{last_error}")


def backfill_put_call(
    as_of: date, days: int, fetch: Fetch, pause: float = 0.25
) -> tuple[list[tuple[str, date, float, str]], int]:
    points, misses = [], 0
    for back in range(days):
        day = as_of - timedelta(days=back)
        if day.weekday() >= 5:
            continue
        try:
            ratios = _ratios_for(day, fetch)
        except FetchError as exc:
            if exc.status is None:
                raise
            misses += 1
            continue
        except ValueError:
            misses += 1
            continue
        points.extend((metric, day, value, "cboe") for metric, value in ratios.items())
        time.sleep(pause)
    return points, misses
