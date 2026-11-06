"""Brokers.

SimulatedBroker   in-memory fills with slippage/commission (used by the backtester)
PaperBroker       SimulatedBroker persisted to state/paper_account.json + trade log CSV
RobinhoodBroker   real orders via the unofficial robin_stocks library (live mode only)
"""
from __future__ import annotations

import csv
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .execution import Order


@dataclass
class Fill:
    ts: str
    symbol: str
    side: str
    qty: float
    price: float
    commission: float
    status: str  # "filled" | "submitted" | "rejected"
    note: str = ""


class Broker(ABC):
    @abstractmethod
    def positions(self) -> dict[str, float]: ...

    @abstractmethod
    def available_cash(self, prices: dict[str, float]) -> float: ...

    @abstractmethod
    def submit(self, order: Order, price: float, ts: str | None = None, *, cost_bps: float | None = None,
               limit_price: float | None = None) -> Fill:
        """price: the price we expect to trade at (quote side for live/paper, open for backtests).
        cost_bps: extra adverse cost applied by simulators. limit_price: used by real brokers."""

    def cancel_stale(self) -> list[str]:
        return []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SimulatedBroker(Broker):
    def __init__(self, cash: float, slippage_bps: float = 5.0, commission: float = 0.0,
                 fractional: bool = True, positions: dict[str, float] | None = None, allow_short: bool = False):
        self.cash = float(cash)
        self.pos: dict[str, float] = dict(positions or {})
        self.slip_bps = slippage_bps
        self.commission = commission
        self.fractional = fractional
        self.allow_short = allow_short
        self.fills: list[Fill] = []

    def positions(self):
        return {s: q for s, q in self.pos.items() if abs(q) > 1e-9}

    def available_cash(self, prices=None):
        return self.cash

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices[s] for s, q in self.pos.items() if q)

    def _q(self, q: float) -> float:
        return math.floor(q * 1e6) / 1e6 if self.fractional else float(math.floor(q))

    def submit(self, order, price, ts=None, *, cost_bps=None, limit_price=None):
        ts = ts or _now()
        slip = (self.slip_bps if cost_bps is None else cost_bps) / 1e4
        if not (math.isfinite(price) and price > 0):
            return self._record(Fill(ts, order.symbol, order.side, 0, price, 0, "rejected", "bad price"))
        if order.side == "buy":
            px = price * (1 + slip)
            qty = order.qty
            if qty * px + self.commission > self.cash:  # scale down to available cash
                qty = self._q(max(self.cash - self.commission, 0) / px)
            if qty <= 0:
                return self._record(Fill(ts, order.symbol, "buy", 0, px, 0, "rejected", "insufficient cash"))
            self.cash -= qty * px + self.commission
            self.pos[order.symbol] = self.pos.get(order.symbol, 0.0) + qty
        else:
            px = price * (1 - slip)
            held = self.pos.get(order.symbol, 0.0)
            qty = order.qty if self.allow_short else min(order.qty, max(held, 0.0))
            if qty <= 0:
                return self._record(Fill(ts, order.symbol, "sell", 0, px, 0, "rejected", "no position"))
            self.cash += qty * px - self.commission
            left = held - qty
            self.pos[order.symbol] = 0.0 if abs(left) < 1e-9 else left
        if abs(self.cash) < 1e-9:
            self.cash = 0.0
        return self._record(Fill(ts, order.symbol, order.side, qty, px, self.commission, "filled"))

    def _record(self, f: Fill) -> Fill:
        self.fills.append(f)
        return f


class PaperBroker(SimulatedBroker):
    """Simulated account that survives restarts. Fills at the live quote +/- slippage."""

    def __init__(self, state_dir: str | Path, starting_cash: float, slippage_bps: float,
                 commission: float, fractional: bool, allow_short: bool = False):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "paper_account.json"
        self.trades_path = self.dir / "paper_trades.csv"
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            super().__init__(st["cash"], slippage_bps, commission, fractional, st.get("positions", {}), allow_short)
            self.starting_cash = st.get("starting_cash", starting_cash)
            self.created_at = st.get("created_at", _now())
        else:
            super().__init__(starting_cash, slippage_bps, commission, fractional, allow_short=allow_short)
            self.starting_cash = starting_cash
            self.created_at = _now()
            self.save()

    def save(self) -> None:
        st = {"cash": self.cash, "positions": self.positions(), "starting_cash": self.starting_cash,
              "created_at": self.created_at, "updated_at": _now()}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        tmp.replace(self.state_path)

    def submit(self, order, price, ts=None, *, cost_bps=None, limit_price=None):
        f = super().submit(order, price, ts, cost_bps=cost_bps)
        new = not self.trades_path.exists()
        with open(self.trades_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(asdict(f)))
            if new:
                w.writeheader()
            w.writerow(asdict(f))
        self.save()
        return f

    def reset(self) -> None:
        for p in (self.state_path, self.trades_path):
            p.unlink(missing_ok=True)


def robinhood_login():
    """Log in with robin_stocks using env vars RH_USERNAME, RH_PASSWORD and optionally
    RH_MFA_CODE (one-time) or RH_TOTP_SECRET (authenticator secret; needs `pip install pyotp`)."""
    try:
        import robin_stocks.robinhood as rh
    except ImportError as e:
        raise RuntimeError("pip install robin_stocks to use Robinhood data or live trading") from e
    user, pw = os.environ.get("RH_USERNAME"), os.environ.get("RH_PASSWORD")
    if not user or not pw:
        raise RuntimeError("set RH_USERNAME and RH_PASSWORD environment variables")
    mfa = os.environ.get("RH_MFA_CODE")
    if not mfa and os.environ.get("RH_TOTP_SECRET"):
        import pyotp
        mfa = pyotp.TOTP(os.environ["RH_TOTP_SECRET"]).now()
    rh.login(user, pw, mfa_code=mfa, store_session=True)
    return rh


class RobinhoodBroker(Broker):
    """Real-money broker. Only touches symbols in the bot's universe and never manages more than
    `max_capital` dollars, whatever else is in the account. Sends marketable LIMIT orders (never naked
    market orders), remembers their ids, and cancels its own unfilled orders on the next cycle.
    Untested against a live account: watch the first runs closely."""

    def __init__(self, universe: list[str], max_capital: float, fractional: bool, state_dir: str | Path):
        self.rh = robinhood_login()
        self.universe = set(universe)
        self.max_capital = max_capital
        self.fractional = fractional
        self.orders_path = Path(state_dir) / "live_orders.json"
        self.orders_path.parent.mkdir(parents=True, exist_ok=True)

    def _ids(self) -> list[str]:
        return json.loads(self.orders_path.read_text()) if self.orders_path.exists() else []

    def _save_ids(self, ids: list[str]) -> None:
        self.orders_path.write_text(json.dumps(ids[-500:]))

    def positions(self):
        holdings = self.rh.account.build_holdings()
        return {s: float(h["quantity"]) for s, h in holdings.items() if s in self.universe and float(h["quantity"]) > 0}

    def available_cash(self, prices):
        bp = float(self.rh.profiles.load_account_profile(info="buying_power"))
        invested = sum(q * prices[s] for s, q in self.positions().items())
        return max(0.0, min(bp, self.max_capital - invested))

    def cancel_stale(self):
        cancelled, keep = [], []
        for oid in self._ids():
            info = self.rh.orders.get_stock_order_info(oid) or {}
            if info.get("state") in ("queued", "unconfirmed", "confirmed", "partially_filled"):
                self.rh.orders.cancel_stock_order(oid)
                cancelled.append(oid)
        self._save_ids(keep)
        return cancelled

    def submit(self, order, price, ts=None, *, cost_bps=None, limit_price=None):
        ts = ts or _now()
        if limit_price is None:
            return Fill(ts, order.symbol, order.side, 0, price, 0, "rejected", "live orders require a limit price")
        qty = round(order.qty, 6) if self.fractional else int(order.qty)
        resp = self.rh.orders.order(order.symbol, qty, order.side, limitPrice=round(limit_price, 2), timeInForce="gfd")
        ok = isinstance(resp, dict) and resp.get("id")
        if ok:
            self._save_ids(self._ids() + [resp["id"]])
        note = resp.get("id") if ok else str(resp)[:200]
        return Fill(ts, order.symbol, order.side, qty, limit_price, 0.0, "submitted" if ok else "rejected", note)
