"""One trading cycle for paper or live mode. Safe to run repeatedly: it rebalances at most once per
period (daily or weekly), otherwise it only marks the book to market.

Each decision is written to state/journal.jsonl: forecasts, alpha weights and ICs, target vs.
current weights, orders and fills. That gives a post-trade record of why every trade happened."""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .backtest import build_panel, period_key
from .broker import Broker, PaperBroker, RobinhoodBroker
from .config import Config
from .costs import fill_cost_bps
from .data import DataProvider, make_provider
from .execution import plan_orders
from .portfolio import Decision, decide
from .risk import RiskManager

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
    day_key: str | None = None
    day_start_equity: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "EngineState":
        if path.exists():
            d = json.loads(path.read_text())
            known = set(cls.__dataclass_fields__) - {"path"}
            return cls(path=path, **{k: v for k, v in d.items() if k in known})
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
    day_pnl: float = 0.0
    action: str = "none"
    orders: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    decision: Decision | None = None
    current_weights: pd.Series | None = None
    signals: pd.DataFrame | None = None


def make_broker(cfg: Config) -> Broker:
    ex, sd = cfg.execution, cfg.path(cfg.general.state_dir)
    if cfg.general.mode == "live":
        if os.environ.get(LIVE_ENV) != "yes":
            raise RuntimeError(f"live mode is locked. Set {LIVE_ENV}=yes to allow real orders.")
        return RobinhoodBroker(cfg.universe.symbols, cfg.risk.max_capital, ex.fractional, sd)
    # paper fills cross the (live or modeled) spread explicitly; per-order slippage+impact is passed in
    return PaperBroker(sd, cfg.risk.starting_cash, cfg.costs.slippage_bps, 0.0, ex.fractional,
                       allow_short=not cfg.portfolio.long_only)


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def signal_table(panel, T, decision: Decision | None, w_now: pd.Series | None) -> pd.DataFrame:
    fc = panel.fc
    df = pd.DataFrame({k: s.loc[T] for k, s in fc.scores.items()})
    df["combined_z"] = fc.combined.loc[T]
    if decision is not None:
        df["exp_ret_bps"] = decision.mu * 1e4
        df["target_w"] = decision.weights
    if w_now is not None:
        df["current_w"] = w_now
    return df.sort_values("combined_z", ascending=False)


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
    if (mode == "live" or cfg.general.data_provider != "csv") and not ignore_hours and not market_open(now):
        rep.action = "market closed; nothing to do"
        return rep

    provider = provider or make_provider(cfg)
    broker = broker or make_broker(cfg)
    risk = RiskManager(cfg.risk)
    syms = cfg.universe.symbols
    if not dry_run:
        cancelled = broker.cancel_stale()
        if cancelled:
            log.info("cancelled %d stale bot orders", len(cancelled))

    today = pd.Timestamp(now.astimezone(NY).date())
    bars = {s: df[df.index < today] for s, df in provider.history(syms).items()}  # completed sessions only
    panel = build_panel(bars, cfg)
    T = panel.C.index[-1]
    quotes = provider.quotes(syms, cfg.costs.half_spread_bps)
    mids = {s: q.mid for s, q in quotes.items()}
    positions = broker.positions()
    cash = broker.available_cash(mids)
    equity = cash + sum(q * mids[s] for s, q in positions.items())
    w_now = pd.Series({s: q * mids[s] / equity for s, q in positions.items()}, dtype=float).reindex(syms, fill_value=0.0) \
        if equity > 0 else pd.Series(0.0, index=syms)
    rep.current_weights = w_now

    day = str(today.date())
    if state.day_key != day:
        state.day_key, state.day_start_equity = day, equity
    state.peak_equity = max(state.peak_equity, equity)
    rep.equity, rep.cash = equity, cash
    rep.drawdown = 1 - equity / state.peak_equity if state.peak_equity else 0.0
    rep.day_pnl = equity / state.day_start_equity - 1 if state.day_start_equity else 0.0

    targets: pd.Series | None = None
    key = str(period_key(today, cfg.portfolio.rebalance))
    if risk.kill_switch(equity, state.peak_equity):
        targets = pd.Series(0.0, index=syms)
        rep.action = f"KILL SWITCH: drawdown {rep.drawdown:.1%} >= {cfg.risk.kill_switch_drawdown:.0%}; flattening"
        state.halted, state.halted_reason = True, f"drawdown {rep.drawdown:.1%} on {day}"
    elif rep.day_pnl <= -cfg.risk.daily_loss_limit:
        rep.action = f"daily loss limit: {rep.day_pnl:+.2%} today; no new trades until tomorrow"
    elif force or state.last_rebalance_key != key:
        d = decide(T, panel.fc, panel.rets, panel.ewma, w_now, cfg, panel.beta)
        rep.decision = d
        if d is None:
            rep.action = "risk model not ready (not enough history)"
        else:
            targets = d.weights
            rep.action = "rebalance" + (" (forced)" if force else "")
    else:
        rep.action = f"holding; already rebalanced this period ({key})"
    rep.signals = signal_table(panel, T, rep.decision, w_now)

    if targets is not None:
        orders = plan_orders(targets, positions, mids, equity, cfg.execution)
        ok, rejected = risk.check_orders(orders, mids, panel.C.iloc[-1].to_dict())
        rep.orders, rep.rejected = ok, rejected
        if dry_run:
            rep.action += " [dry run: no orders sent]"
        else:
            if mode == "live" and ok and confirm is not None and not confirm(ok):
                rep.action += " [cancelled by user]"
                return rep
            buf = cfg.execution.limit_buffer_bps / 1e4
            for o in ok:
                q = quotes[o.symbol]
                px = q.ask if o.side == "buy" else q.bid  # cross the spread explicitly
                bps = fill_cost_bps(o.notional, panel.adv.at[T, o.symbol], panel.ewma.at[T, o.symbol], cfg.costs,
                                    crossing_spread=False)
                limit = q.ask * (1 + buf) if o.side == "buy" else q.bid * (1 - buf)
                rep.fills.append(broker.submit(o, px, cost_bps=bps, limit_price=limit))
            if not state.halted:
                state.last_rebalance_key, state.last_rebalance_at = key, rep.ts

    if not dry_run:
        positions = broker.positions()
        cash = broker.available_cash(mids)
        rep.equity, rep.cash = cash + sum(q * mids[s] for s, q in positions.items()), cash
        rep.drawdown = 1 - rep.equity / state.peak_equity if state.peak_equity else 0.0
        state.save()
        _append_csv(state_dir / f"{mode}_equity.csv", {
            "ts": rep.ts, "equity": round(rep.equity, 2), "cash": round(cash, 2), "peak": round(state.peak_equity, 2),
            "drawdown": round(rep.drawdown, 4), "day_pnl": round(rep.day_pnl, 4), "action": rep.action})
        if rep.decision is not None or rep.fills:
            entry = {"ts": rep.ts, "mode": mode, "signal_date": str(T.date()), "action": rep.action,
                     "equity": rep.equity,
                     "combined_ic": rep.decision.combined_ic if rep.decision else None,
                     "ex_ante_vol": rep.decision.ex_ante_vol if rep.decision else None,
                     "alpha_weights": rep.decision.alpha_weights.round(4).to_dict() if rep.decision else None,
                     "target_w": rep.decision.weights[rep.decision.weights != 0].round(4).to_dict() if rep.decision else None,
                     "current_w": w_now[w_now != 0].round(4).to_dict(),
                     "exp_ret_bps": (rep.decision.mu * 1e4).round(2).to_dict() if rep.decision else None,
                     "rejected": [(o.symbol, why) for o, why in rep.rejected],
                     "fills": [asdict(f) for f in rep.fills]}
            with open(state_dir / "journal.jsonl", "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
    return rep
