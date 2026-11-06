"""Ingest market data fetched by Claude through the Robinhood connector (for running inside Claude Code,
where the sandbox can't reach market-data hosts directly).

Bars:   the JSON saved by Claude Code when a get_equity_historicals result is too large to show inline
        ({"data": {"results": [{"symbol", "interval", "bars": [...]}]}}). Parsed exactly; nothing is retyped.
Quotes: one line per symbol, written by Claude from get_equity_quotes:
            SYMBOL BID ASK LAST QUOTE_TIME_UTC PREV_CLOSE PREV_CLOSE_DATE
        Because these are transcribed, every line is cross-checked: PREV_CLOSE must equal the exact
        close in the bar data for PREV_CLOSE_DATE, bid <= ask, the spread must be sane, and LAST must sit
        near the bid/ask and within 10% of the previous close. Any failure rejects the whole file.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import COLS


class IngestError(ValueError):
    pass


def ingest_bars(paths: list[str | Path], data_dir: str | Path, symbols: list[str]) -> dict[str, tuple[str, int]]:
    out: dict[str, pd.DataFrame] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        results = doc.get("data", {}).get("results") if isinstance(doc, dict) else None
        if not results:
            raise IngestError(f"{p}: not a get_equity_historicals result")
        for r in results:
            if r.get("interval") != "day":
                raise IngestError(f"{p}: {r.get('symbol')} has interval {r.get('interval')!r}, need 'day'")
            rows = [b for b in r["bars"] if not b.get("interpolated")]
            df = pd.DataFrame({
                "open": [float(b["open_price"]) for b in rows], "high": [float(b["high_price"]) for b in rows],
                "low": [float(b["low_price"]) for b in rows], "close": [float(b["close_price"]) for b in rows],
                "volume": [float(b["volume"]) for b in rows],
            }, index=pd.to_datetime([b["begins_at"][:10] for b in rows]))
            out[r["symbol"]] = df[~df.index.duplicated(keep="last")].sort_index()
    missing = [s for s in symbols if s not in out]
    if missing:
        raise IngestError(f"bars missing for {missing}")
    last = {s: out[s].index[-1] for s in symbols}
    if len(set(last.values())) != 1:
        raise IngestError(f"symbols end on different dates: {sorted({str(d.date()) for d in last.values()})}")
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for s in symbols:
        df = out[s]
        if len(df) < 300 or (df[["open", "close"]] <= 0).any().any():
            raise IngestError(f"{s}: suspicious bar data ({len(df)} bars)")
        df = df[COLS].copy()
        df["volume"] = df["volume"].round().astype("int64")
        df.to_csv(d / f"{s}.csv", index_label="date", float_format="%.6f")
    return {s: (str(out[s].index[-1].date()), len(out[s])) for s in symbols}


def ingest_quotes(path: str | Path, data_dir: str | Path, symbols: list[str], *, now: datetime | None = None,
                  max_age_minutes: float = 30.0) -> dict:
    now = now or datetime.now(timezone.utc)
    quotes, errors = {}, []
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            errors.append(f"line {ln}: expected 7 fields, got {len(parts)}")
            continue
        sym, bid, ask, last, ts, prev, prev_date = parts
        try:
            bid, ask, last, prev = float(bid), float(ask), float(last), float(prev)
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError as e:
            errors.append(f"line {ln}: {e}")
            continue
        csv = Path(data_dir) / f"{sym}.csv"
        if not csv.exists():
            errors.append(f"{sym}: no bar data")
            continue
        bars = pd.read_csv(csv, index_col="date", parse_dates=True)
        ref = bars["close"].get(pd.Timestamp(prev_date))
        if ref is None or (isinstance(ref, float) and math.isnan(ref)):
            errors.append(f"{sym}: no bar for prev-close date {prev_date}; ingest fresh bars first")
            continue
        if abs(prev / ref - 1) > 1e-6:
            errors.append(f"{sym}: prev close {prev} != bar close {ref} on {prev_date} (transcription error?)")
        if not (0 < bid <= ask):
            errors.append(f"{sym}: bad book bid={bid} ask={ask}")
        elif ask / bid - 1 > 0.01:
            errors.append(f"{sym}: spread {ask / bid - 1:.2%} too wide")
        elif not (bid * 0.995 <= last <= ask * 1.005):
            errors.append(f"{sym}: last {last} outside book {bid}/{ask}")
        if abs(last / prev - 1) > 0.10:
            errors.append(f"{sym}: last {last} is {last / prev - 1:+.1%} from prev close")
        age = (now - t).total_seconds() / 60
        if age > max_age_minutes or age < -5:
            errors.append(f"{sym}: quote time {ts} is {age:.0f} min old")
        quotes[sym] = {"bid": bid, "ask": ask, "last": last, "time": ts, "prev_close": prev, "prev_close_date": prev_date}
    missing = [s for s in symbols if s not in quotes]
    if missing:
        errors.append(f"missing quotes for {missing}")
    if errors:
        raise IngestError("quote file rejected:\n  " + "\n  ".join(errors))
    doc = {"ingested_at": now.isoformat(timespec="seconds"), "source": "robinhood connector (get_equity_quotes)",
           "quotes": {s: quotes[s] for s in symbols}}
    (Path(data_dir) / "quotes.json").write_text(json.dumps(doc, indent=1))
    return doc
