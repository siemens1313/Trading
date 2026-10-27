"""Daily closes: Yahoo's chart API, falling back to Stooq's CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timezone
from typing import Callable
from urllib.parse import quote

from .http import FetchError
from .util import parse_day

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2y&interval=1d"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_YAHOO_SYMBOLS = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX"}
_STOOQ_SYMBOLS = {"SPX": "^spx", "NDX": "^ndx"}


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
    errors = []
    try:
        payload = json.loads(fetch(YAHOO_URL.format(symbol=quote(_YAHOO_SYMBOLS.get(symbol, symbol)))))
        return [(metric, day, value, "yahoo") for day, value in parse_yahoo_chart(payload)]
    except Exception as exc:  # format drift or network: try the next source
        errors.append(f"yahoo: {exc}")
    try:
        text = fetch(STOOQ_URL.format(symbol=_STOOQ_SYMBOLS.get(symbol, f"{symbol.lower()}.us"))).decode()
        return [(metric, day, value, "stooq") for day, value in parse_stooq_csv(text)]
    except Exception as exc:
        errors.append(f"stooq: {exc}")
    raise FetchError("; ".join(errors))
