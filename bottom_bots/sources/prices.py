"""Daily closes: Cboe's chart history, then Yahoo's chart API, then Stooq's CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timezone
from typing import Callable
from urllib.parse import quote

from .http import FetchError
from .util import parse_day

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/{symbol}.json"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2y&interval=1d"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_CBOE_SYMBOLS = {"SPX": "_SPX", "NDX": "_NDX", "VIX": "_VIX"}
_YAHOO_SYMBOLS = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX"}
_STOOQ_SYMBOLS = {"SPX": "^spx", "NDX": "^ndx"}


def parse_cboe_history(payload: dict) -> list[tuple[date, float]]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("no 'data' list in the Cboe chart response")
    out = []
    for row in rows:
        try:
            out.append((parse_day(str(row["date"])[:10]), float(row["close"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise ValueError("no closes in the Cboe chart response")
    return out


def parse_yahoo_chart(payload: dict) -> list[tuple[date, float]]:
    result = payload["chart"]["result"][0]
    offset = int(result.get("meta", {}).get("gmtoffset", 0))
    closes = result["indicators"]["quote"][0]["close"]
    out = []
    for stamp, close in zip(result["timestamp"], closes):
        if close is None:
            continue
        out.append((datetime.fromtimestamp(stamp + offset, tz=timezone.utc).date(), float(close)))
    return out


def parse_stooq_csv(text: str) -> list[tuple[date, float]]:
    if text.lstrip().startswith("<"):
        raise ValueError("Stooq returned an HTML page instead of CSV (blocked, or it wants an API key)")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "Close" not in reader.fieldnames:
        raise ValueError(f"unexpected Stooq response: {text[:80]!r}")
    out = []
    for row in reader:
        try:
            out.append((parse_day(row["Date"]), float(row["Close"])))
        except (TypeError, ValueError):
            continue
    return out


def fetch_closes(symbol: str, fetch: Callable[[str], bytes]) -> list[tuple[str, date, float, str]]:
    metric = f"close:{symbol.upper()}"
    sources = (
        ("cboe", CBOE_URL.format(symbol=_CBOE_SYMBOLS.get(symbol, symbol)), lambda raw: parse_cboe_history(json.loads(raw))),
        ("yahoo", YAHOO_URL.format(symbol=quote(_YAHOO_SYMBOLS.get(symbol, symbol))), lambda raw: parse_yahoo_chart(json.loads(raw))),
        ("stooq", STOOQ_URL.format(symbol=_STOOQ_SYMBOLS.get(symbol, f"{symbol.lower()}.us")), lambda raw: parse_stooq_csv(raw.decode())),
    )
    errors = []
    for name, url, parse in sources:
        try:
            return [(metric, day, value, name) for day, value in parse(fetch(url))]
        except Exception as exc:  # format drift, blocking or network: try the next source
            errors.append(f"{name}: {exc}")
    raise FetchError("; ".join(errors))
