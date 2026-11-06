"""Market data providers. All return daily OHLCV DataFrames indexed by date with columns open/high/low/close/volume."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    @abstractmethod
    def history(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Daily bars for each symbol, oldest first."""

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Most recent tradable price. Default: last daily close."""
        h = self.history(symbols)
        return {s: float(df["close"].iloc[-1]) for s, df in h.items()}


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


def make_provider(cfg) -> DataProvider:
    kind = cfg.general.data_provider
    if kind == "csv":
        return CSVProvider(cfg.path(cfg.general.data_dir))
    if kind == "yahoo":
        return YahooProvider()
    if kind == "robinhood":
        return RobinhoodProvider()
    raise ValueError(kind)


def align(bars: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """Wide DataFrame (dates x symbols) of one field, restricted to dates where every symbol has data."""
    wide = pd.DataFrame({s: df[field] for s, df in bars.items()})
    return wide.dropna(how="any")
