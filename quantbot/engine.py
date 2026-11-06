"""One trading cycle for paper or live mode. Safe to run repeatedly (e.g. every 15 min or once a day):
it only rebalances once per rebalance period, and otherwise just marks the account to market."""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import period_key
from .broker import Broker, PaperBroker, RobinhoodBroker
from .config import Config
from .data import DataProvider, align, make_provider
from .execution import plan_orders
from .risk import RiskManager
from .strategy import compute_targets, explain

log = logging.getLogger("quantbot")
NY = ZoneInfo("America/New_York")
LIVE_ENV = "QUANTBOT_ALLOW_LIVE"


def market_open(now: datetime) -> bool:
    """Regular session check (Mon-Fri 9:30-16:00 ET). Exchange holidays are not modeled."""
    t = now.astimezone(NY)
    return t.weekday() < 5 and (9, 30) <= (t.hour, t.minute) < (16, 0)


@dataclass
class EngineState:
    path: Path
    last_rebalance_key: str | None = None
    last_rebalance_at: str | None = None
    peak_equity: float = 0.0
    halted: bool = False
    halted_reason: str = ""

    @classmethod
    def load(cls, path: Path) -> "EngineState":
        if path.exists():
            d = json.loads(path.read_text())
            return cls(path=path, **d)
        return cls(path=path)

    def save(self) -> None:
        d = {k: v for k, v in self.__dict__.items() if k != "path"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(self.path)


@dataclass
class CycleReport:
    ts: str
    mode: str
    equity: float = 0.0
    cash: float = 0.0
    drawdown: float = 0.0
    action: str = "none"
    orders: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    signals: pd.DataFrame | None = None


def make_broker(cfg: Config) -> Broker:
    if cfg.general.mode == "live":
        if os.environ.get(LIVE_ENV) != "yes":
            raise RuntimeError(f"live mode is locked. Set {LIVE_ENV}=yes to allow real orders.")
        return RobinhoodBroker(cfg.universe.symbols, cfg.risk.max_capital, cfg.execution.fractional)
    ex = cfg.execution
    return PaperBroker(cfg.path(cfg.general.state_dir), cfg.risk.starting_cash, ex.slippage_bps,
                       ex.commission_per_order, ex.fractional)


def _append_equity(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def run_cycle(cfg: Config, broker: Broker | None = None, provider: DataProvider | None = None, *,
              force: bool = False, dry_run: bool = False, ignore_hours: bool = False,
              confirm=None, now: datetime | None = None) -> CycleReport:
    now = now or datetime.now(NY)
    mode = cfg.general.mode
    rep = CycleReport(ts=now.isoformat(timespec="seconds"), mode=mode)
    state_dir = cfg.path(cfg.general.state_dir)
    state = EngineState.load(state_dir / "engine_state.json")

    if state.halted:
        rep.action = f"halted ({state.halted_reason}). Review, then run `python -m quantbot resume`."
        return rep
    live_data = cfg.general.data_provider != "csv"
    if (mode == "live" or live_data) and not ignore_hours and not market_open(now):
        rep.action = "market closed; nothing to do"
        return rep

    provider = provider or make_provider(cfg)
    broker = broker or make_broker(cfg)
    risk = RiskManager(cfg.risk)
    syms = cfg.universe.symbols

    bars = provider.history(syms)
    closes = align(bars, "close")
    today = pd.Timestamp(now.astimezone(NY).date())
    closes = closes[closes.index < today]  # completed sessions only; drops any partial bar for today
    prices = provider.latest_prices(syms)
    positions = broker.positions()
    cash = broker.available_cash(prices)
    equity = cash + sum(q * prices[s] for s, q in positions.items())
    state.peak_equity = max(state.peak_equity, equity)
    rep.equity, rep.cash = equity, cash
    rep.drawdown = 1 - equity / state.peak_equity if state.peak_equity else 0.0
    rep.signals = explain(closes, cfg.strategy)

    targets = None
    key = str(period_key(today, cfg.strategy.rebalance))
    if risk.kill_switch(equity, state.peak_equity):
        targets = pd.Series(0.0, index=syms)
        rep.action = f"KILL SWITCH: drawdown {rep.drawdown:.1%} >= {cfg.risk.kill_switch_drawdown:.0%}; flattening"
        state.halted, state.halted_reason = True, f"drawdown {rep.drawdown:.1%} on {today.date()}"
    elif force or state.last_rebalance_key != key:
        targets = compute_targets(closes, cfg.strategy)
        if targets is None:
            rep.action = "not enough history for a signal"
        else:
            rep.action = "rebalance" + (" (forced)" if force else "")
    else:
        rep.action = f"holding; already rebalanced this period ({key})"

    if targets is not None:
        orders = plan_orders(targets, positions, prices, equity, cfg.execution)
        last_close = closes.iloc[-1].to_dict() if len(closes) else {}
        ok, rejected = risk.check_orders(orders, prices, last_close)
        rep.orders, rep.rejected = ok, rejected
        if dry_run:
            rep.action += " [dry run: no orders sent]"
        else:
            if mode == "live" and ok and confirm is not None and not confirm(ok):
                rep.action += " [cancelled by user]"
                return rep
            for o in ok:
                rep.fills.append(broker.submit(o, prices[o.symbol]))
            if not state.halted:
                state.last_rebalance_key, state.last_rebalance_at = key, rep.ts

    if not dry_run:
        positions = broker.positions()
        cash = broker.available_cash(prices)
        rep.equity = cash + sum(q * prices[s] for s, q in positions.items())
        rep.cash = cash
        rep.drawdown = 1 - rep.equity / state.peak_equity if state.peak_equity else 0.0
        state.save()
        _append_equity(state_dir / f"{mode}_equity.csv", {
            "ts": rep.ts, "equity": round(rep.equity, 2), "cash": round(cash, 2),
            "peak": round(state.peak_equity, 2), "drawdown": round(rep.drawdown, 4), "action": rep.action,
        })
    return rep
