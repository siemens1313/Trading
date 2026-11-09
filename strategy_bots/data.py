"""Bars for the two bots: daily panels and intraday sessions.

Sources: CSV files (date,open,high,low,close,volume, the same format quantbot uses), the JSON that Claude
Code saves from the Robinhood connector's get_equity_historicals (parsed exactly, never retyped), and
seeded synthetic data for tests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

FIELDS = ("open", "high", "low", "close", "volume")
NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN, SESSION_CLOSE = 9 * 60 + 30, 16 * 60  # minutes after midnight, New York


class DataError(ValueError):
    pass


@dataclass
class Panel:
    """Daily bars for several symbols on shared dates; each field has shape (days, symbols)."""

    symbols: list[str]
    dates: np.ndarray  # datetime64[D]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)

    def col(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def head(self, n: int) -> Panel:
        """The first n days: everything the bot knew at the close of day n - 1."""
        return Panel(self.symbols, self.dates[:n], *(getattr(self, f)[:n] for f in FIELDS))

    def select(self, symbols: list[str]) -> Panel:
        idx = [self.col(s) for s in symbols]
        return Panel(list(symbols), self.dates, *(getattr(self, f)[:, idx] for f in FIELDS))


@dataclass
class Session:
    """One regular trading session of intraday bars for one symbol."""

    day: date
    minutes: np.ndarray  # bar start, minutes after midnight New York
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.minutes)


def load_daily_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise DataError(f"{path}: no rows")
    out = {"dates": np.array([r["date"][:10] for r in rows], dtype="datetime64[D]")}
    for field in FIELDS:
        out[field] = np.array([float(r[field]) for r in rows])
    order = np.argsort(out["dates"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def load_panel(data_dir: Path, symbols: list[str] | None = None) -> Panel:
    """Daily CSVs joined on the dates every symbol has."""
    data_dir = Path(data_dir)
    if symbols is None:
        symbols = sorted(p.stem for p in data_dir.glob("*.csv"))
    if not symbols:
        raise DataError(f"no daily CSV files in {data_dir}")
    series = {}
    for s in symbols:
        path = data_dir / f"{s}.csv"
        if not path.exists():
            raise DataError(f"{path} not found")
        series[s] = load_daily_csv(path)
    common = series[symbols[0]]["dates"]
    for s in symbols[1:]:
        common = np.intersect1d(common, series[s]["dates"])
    if len(common) == 0:
        raise DataError("the symbols share no dates")
    fields = []
    for field in FIELDS:
        cols = []
        for s in symbols:
            keep = np.isin(series[s]["dates"], common)
            cols.append(series[s][field][keep])
        fields.append(np.column_stack(cols))
    return Panel(list(symbols), common, *fields)


def load_sessions(path: Path) -> list[Session]:
    """Intraday CSV (datetime in New York time, open, high, low, close, volume) as regular sessions."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    by_day: dict[date, list] = {}
    for r in rows:
        stamp = datetime.fromisoformat(r["datetime"])
        minute = stamp.hour * 60 + stamp.minute
        if SESSION_OPEN <= minute < SESSION_CLOSE:
            by_day.setdefault(stamp.date(), []).append((minute, *(float(r[f]) for f in FIELDS)))
    sessions = []
    for day in sorted(by_day):
        bars = np.array(sorted(by_day[day]))
        sessions.append(Session(day, bars[:, 0].astype(int), *(bars[:, i] for i in range(1, 6))))
    return sessions


def _write_rows(path: Path, key: str, rows: dict[str, tuple]) -> None:
    """Merge rows into a CSV keyed by date or datetime; new values replace old ones."""
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r[key], tuple(float(r[field]) for field in FIELDS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key, *FIELDS])
        for k in sorted(rows):
            o, h, l, c, v = rows[k]
            w.writerow([k, f"{o:.6f}", f"{h:.6f}", f"{l:.6f}", f"{c:.6f}", int(round(v))])


def ingest_robinhood(paths: list[Path], data_dir: Path) -> dict[str, str]:
    """Write bars from saved get_equity_historicals results to data_dir/daily and data_dir/intraday."""
    written = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        results = doc.get("data", {}).get("results") if isinstance(doc, dict) else None
        if not results:
            raise DataError(f"{p}: not a get_equity_historicals result")
        for r in results:
            symbol, interval = r["symbol"], r["interval"]
            rows = {}
            for b in r["bars"]:
                if b.get("interpolated"):
                    continue
                values = tuple(float(b[k]) for k in ("open_price", "high_price", "low_price", "close_price", "volume"))
                if min(values[:4]) <= 0:
                    raise DataError(f"{p}: {symbol} has a non-positive price at {b['begins_at']}")
                stamp = datetime.fromisoformat(b["begins_at"].replace("Z", "+00:00"))
                if interval == "day":
                    rows[b["begins_at"][:10]] = values
                else:
                    rows[stamp.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M")] = values
            if interval == "day":
                path, key = Path(data_dir) / "daily" / f"{symbol}.csv", "date"
            else:
                path, key = Path(data_dir) / "intraday" / f"{symbol}_{interval}.csv", "datetime"
            _write_rows(path, key, rows)
            written[str(path)] = f"{len(rows)} bars"
    return written


def synthetic_panel(
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    days: int = 900,
    seed: int = 7,
    start: str = "2020-01-02",
    drift: float = 0.08,
    vol: float = 0.25,
) -> Panel:
    """Correlated random-walk stocks with plausible OHLCV, for tests."""
    rng = np.random.default_rng(seed)
    n = len(symbols)
    dates = np.busday_offset(np.datetime64(start, "D"), np.arange(days), roll="forward")
    market = rng.normal(drift / 252, vol * 0.6 / np.sqrt(252), (days, 1))
    own = rng.normal(0, vol * 0.8 / np.sqrt(252), (days, n))
    close = 100 * np.exp(np.cumsum(market + own, axis=0))
    gap = rng.normal(0, 0.003, (days, n))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(gap)
    span = np.abs(rng.normal(0, 0.008, (days, n)))
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    volume = rng.lognormal(14, 0.3, (days, n))
    return Panel(list(symbols), dates, open_, high, low, close, volume)


def synthetic_sessions(days: int = 12, bars: int = 78, seed: int = 3, price: float = 100.0) -> list[Session]:
    """Random-walk sessions with the U-shaped volume of a real trading day, for tests."""
    rng = np.random.default_rng(seed)
    minutes = SESSION_OPEN + 5 * np.arange(bars)
    shape = 1.0 + 2.5 * np.exp(-np.arange(bars) / 6) + 2.0 * np.exp(-(bars - 1 - np.arange(bars)) / 6)
    sessions = []
    day = np.datetime64("2026-01-05", "D")
    for i in range(days):
        steps = rng.normal(0, 0.0015, bars)
        close = price * np.exp(np.cumsum(steps))
        open_ = np.concatenate([[price], close[:-1]])
        wick = np.abs(rng.normal(0, 0.0008, bars))
        high = np.maximum(open_, close) * (1 + wick)
        low = np.minimum(open_, close) * (1 - wick)
        volume = shape * rng.lognormal(10, 0.2, bars)
        when = np.busday_offset(day, i, roll="forward").astype(date)
        sessions.append(Session(when, minutes.copy(), open_, high, low, close, volume))
        price = close[-1]
    return sessions
