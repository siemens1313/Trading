"""Market data providers. All return daily OHLCV DataFrames indexed by date with columns open/high/low/close/volume."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class Quote:
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


class DataProvider(ABC):
    @abstractmethod
    def history(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Daily bars for each symbol, oldest first."""

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Most recent tradable price. Default: last daily close."""
        h = self.history(symbols)
        return {s: float(df["close"].iloc[-1]) for s, df in h.items()}

    def quotes(self, symbols: list[str], half_spread_bps: float) -> dict[str, Quote]:
        """Bid/ask. Providers without a live book get a modeled spread around the last price."""
        k = half_spread_bps / 1e4
        return {s: Quote(p * (1 - k), p * (1 + k), p) for s, p in self.latest_prices(symbols).items()}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[COLS].astype(float)
    return df.dropna(subset=["open", "close"])


class CSVProvider(DataProvider):
    """Reads data/<SYMBOL>.csv with columns date,open,high,low,close,volume."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)

    def history(self, symbols):
        out = {}
        for s in symbols:
            p = self.dir / f"{s}.csv"
            if not p.exists():
                raise FileNotFoundError(f"missing {p}; run `python -m quantbot update-data` first")
            out[s] = _clean(pd.read_csv(p, index_col="date", parse_dates=True))
        return out

    def save(self, bars: dict[str, pd.DataFrame]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for s, df in bars.items():
            df[COLS].to_csv(self.dir / f"{s}.csv", index_label="date", float_format="%.6f")


class YahooProvider(DataProvider):
    """Free daily data via yfinance (pip install yfinance). Prices are split/dividend adjusted."""

    def __init__(self, period: str = "5y"):
        import yfinance  # noqa: F401  (fail fast with a clear ImportError)
        self.period = period

    def history(self, symbols):
        import yfinance as yf
        raw = yf.download(symbols, period=self.period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        out = {}
        for s in symbols:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.rename(columns=str.lower)
            out[s] = _clean(df)
        return out

    def latest_prices(self, symbols):
        import yfinance as yf
        prices = {}
        for s in symbols:
            fi = yf.Ticker(s).fast_info
            prices[s] = float(fi["last_price"])
        return prices


class RobinhoodProvider(DataProvider):
    """Data via the unofficial robin_stocks library. Requires login (see broker.robinhood_login)."""

    def __init__(self, span: str = "5year"):
        from .broker import robinhood_login
        self.rs = robinhood_login()
        self.span = span

    def history(self, symbols):
        rows = self.rs.stocks.get_stock_historicals(symbols, interval="day", span=self.span, bounds="regular")
        df = pd.DataFrame(rows)
        out = {}
        for s, g in df.groupby("symbol"):
            g = g[g.get("interpolated", False) != True]  # noqa: E712
            bars = pd.DataFrame({
                "open": g["open_price"].astype(float).values,
                "high": g["high_price"].astype(float).values,
                "low": g["low_price"].astype(float).values,
                "close": g["close_price"].astype(float).values,
                "volume": g["volume"].astype(float).values,
            }, index=pd.to_datetime(g["begins_at"]).values)
            out[s] = _clean(bars)
        return out

    def latest_prices(self, symbols):
        px = self.rs.stocks.get_latest_price(symbols, includeExtendedHours=False)
        return {s: float(p) for s, p in zip(symbols, px)}

    def quotes(self, symbols, half_spread_bps):
        out = {}
        for q in self.rs.stocks.get_quotes(symbols):
            if not q:
                continue
            last = float(q["last_trade_price"])
            bid, ask = float(q.get("bid_price") or 0), float(q.get("ask_price") or 0)
            if not (0 < bid <= ask) or ask / bid - 1 > 0.01:  # missing or absurd book -> model it
                k = half_spread_bps / 1e4
                bid, ask = last * (1 - k), last * (1 + k)
            out[q["symbol"]] = Quote(bid, ask, last)
        return out


class ConnectorFileProvider(CSVProvider):
    """Bars from data/*.csv and live quotes from data/quotes.json, both written by `quantbot ingest-*`
    from data Claude fetched via the Robinhood connector. Refuses stale quotes."""

    def __init__(self, data_dir, max_age_minutes: float = 30.0):
        super().__init__(data_dir)
        self.max_age = max_age_minutes

    def _load(self, symbols):
        import json
        from datetime import datetime, timezone
        p = self.dir / "quotes.json"
        if not p.exists():
            raise RuntimeError("no data/quotes.json; run `python -m quantbot ingest-quotes` first")
        doc = json.loads(p.read_text())
        now = datetime.now(timezone.utc)
        out = {}
        for s in symbols:
            q = doc["quotes"].get(s)
            if q is None:
                raise RuntimeError(f"no quote for {s}")
            age = (now - datetime.fromisoformat(q["time"].replace("Z", "+00:00"))).total_seconds() / 60
            if age > self.max_age:
                raise RuntimeError(f"quote for {s} is {age:.0f} min old (max {self.max_age:.0f}); refetch quotes")
            out[s] = q
        return out

    def latest_prices(self, symbols):
        return {s: q["last"] for s, q in self._load(symbols).items()}

    def quotes(self, symbols, half_spread_bps):
        return {s: Quote(q["bid"], q["ask"], q["last"]) for s, q in self._load(symbols).items()}


def make_provider(cfg) -> DataProvider:
    kind = cfg.general.data_provider
    if kind == "csv":
        return CSVProvider(cfg.path(cfg.general.data_dir))
    if kind == "connector":
        return ConnectorFileProvider(cfg.path(cfg.general.data_dir), cfg.general.max_quote_age_minutes)
    if kind == "yahoo":
        return YahooProvider()
    if kind == "robinhood":
        return RobinhoodProvider()
    raise ValueError(kind)


def align(bars: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Wide DataFrame (dates x symbols) of one field, restricted to dates where every symbol has data."""
    wide = pd.DataFrame({s: df[field] for s, df in bars.items()})
    return wide.dropna(how="any")
