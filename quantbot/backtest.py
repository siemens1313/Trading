"""Daily event-driven backtest.

Timing (no lookahead): signals use closes up to day t. If day t is the last trading day of a
rebalance period, orders are sized at t's close and filled at the OPEN of day t+1, with slippage.
The live engine does the same thing: on the first run of a new period it uses completed bars only
and trades at the current quote.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .broker import SimulatedBroker
from .config import Config
from .data import align
from .execution import Order, plan_orders
from .risk import RiskManager
from .strategy import compute_targets, required_history


def period_key(ts: pd.Timestamp, freq: str):
    if freq == "daily":
        return ts.date()
    if freq == "weekly":
        iso = ts.isocalendar()
        return (iso[0], iso[1])
    return (ts.year, ts.month)


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict
    halted_on: pd.Timestamp | None


def metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * math.sqrt(252)
    downside = r[r < 0].std() * math.sqrt(252)
    dd = equity / equity.cummax() - 1
    m = {
        "start": equity.index[0].date(), "end": equity.index[-1].date(),
        "total_return": total, "cagr": cagr, "ann_vol": vol,
        "sharpe": (r.mean() * 252) / vol if vol > 0 else np.nan,
        "sortino": (r.mean() * 252) / downside if downside > 0 else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "best_day": r.max(), "worst_day": r.min(),
    }
    if trades is not None:
        m["trades"] = int(len(trades))
        notional = (trades["qty"] * trades["price"]).sum() if len(trades) else 0.0
        m["turnover_per_year"] = notional / equity.mean() / years if years > 0 else np.nan
    return m


def run_backtest(bars: dict[str, pd.DataFrame], cfg: Config,
                 start: str | None = None, end: str | None = None) -> BacktestResult:
    opens, closes = align(bars, "open"), align(bars, "close")
    idx = closes.index.intersection(opens.index)
    opens, closes = opens.loc[idx], closes.loc[idx]
    if end:
        opens, closes = opens.loc[:end], closes.loc[:end]
    # Never score the warm-up period where no signal can exist yet.
    warm = closes.index[min(required_history(cfg.strategy) - 1, len(closes) - 1)]
    start_ts = max(pd.Timestamp(start), warm) if start else warm

    p, ex, rk = cfg.strategy, cfg.execution, cfg.risk
    broker = SimulatedBroker(rk.starting_cash, ex.slippage_bps, ex.commission_per_order, ex.fractional)
    risk = RiskManager(rk)
    dates = closes.index
    pending: pd.Series | None = None
    peak, halted_on = 0.0, None
    eq_rows, w_rows, trade_rows = {}, {}, []

    for i, t in enumerate(dates):
        # 1) Execute yesterday's decision at today's open.
        if pending is not None and halted_on is None:
            o_px = opens.loc[t].to_dict()
            equity_open = broker.equity(o_px)
            for o in plan_orders(pending, broker.positions(), o_px, equity_open, ex):
                f = broker.submit(o, o_px[o.symbol], ts=str(t.date()))
                if f.status == "filled":
                    trade_rows.append({"date": t, "symbol": f.symbol, "side": f.side, "qty": f.qty, "price": f.price})
            pending = None

        # 2) Mark to market at the close; kill switch.
        c_px = closes.loc[t].to_dict()
        if t < start_ts:
            continue
        eq = broker.equity(c_px)
        peak = max(peak, eq)
        if halted_on is None and risk.kill_switch(eq, peak):
            for s, q in list(broker.positions().items()):
                broker.submit(Order(s, "sell", q, c_px[s]), c_px[s], ts=str(t.date()))
                trade_rows.append({"date": t, "symbol": s, "side": "sell", "qty": q, "price": c_px[s]})
            eq = broker.equity(c_px)
            halted_on = t
        eq_rows[t] = eq
        w_rows[t] = pd.Series({s: q * c_px[s] / eq for s, q in broker.positions().items()} if eq > 0 else {},
                              dtype=float).reindex(closes.columns, fill_value=0.0)

        # 3) Decide: rebalance after the last close of each period (next date is in a new period).
        if halted_on is None and i + 1 < len(dates):
            nxt = dates[i + 1]
            first = pending is None and not trade_rows and not broker.positions()
            if first or period_key(nxt, p.rebalance) != period_key(t, p.rebalance):
                tgt = compute_targets(closes.iloc[: i + 1], p)
                if tgt is not None:
                    pending = tgt

    equity = pd.Series(eq_rows, name="strategy")
    if equity.empty:
        raise ValueError("no backtest dates in range (check --start/--end and data length)")
    bench_px = closes[cfg.universe.benchmark].loc[equity.index]
    benchmark = (bench_px / bench_px.iloc[0] * rk.starting_cash).rename(cfg.universe.benchmark)
    trades = pd.DataFrame(trade_rows, columns=["date", "symbol", "side", "qty", "price"])
    weights = pd.DataFrame(w_rows).T
    return BacktestResult(equity, benchmark, weights, trades, metrics(equity, trades), metrics(benchmark), halted_on)


def format_report(res: BacktestResult, bench_name: str) -> str:
    rows = [
        ("Period", f"{res.metrics['start']} -> {res.metrics['end']}", ""),
        ("Total return", f"{res.metrics['total_return']:+.1%}", f"{res.benchmark_metrics['total_return']:+.1%}"),
        ("CAGR", f"{res.metrics['cagr']:+.2%}", f"{res.benchmark_metrics['cagr']:+.2%}"),
        ("Ann. volatility", f"{res.metrics['ann_vol']:.2%}", f"{res.benchmark_metrics['ann_vol']:.2%}"),
        ("Sharpe (rf=0)", f"{res.metrics['sharpe']:.2f}", f"{res.benchmark_metrics['sharpe']:.2f}"),
        ("Sortino", f"{res.metrics['sortino']:.2f}", f"{res.benchmark_metrics['sortino']:.2f}"),
        ("Max drawdown", f"{res.metrics['max_drawdown']:.1%}", f"{res.benchmark_metrics['max_drawdown']:.1%}"),
        ("Calmar", f"{res.metrics['calmar']:.2f}", f"{res.benchmark_metrics['calmar']:.2f}"),
        ("Trades", f"{res.metrics['trades']}", ""),
        ("Turnover / year", f"{res.metrics['turnover_per_year']:.1f}x", ""),
        ("Avg. invested", f"{res.weights.sum(axis=1).mean():.0%}", "100%"),
    ]
    w = max(len(r[0]) for r in rows)
    out = [f"{'':<{w}}  {'Strategy':>24}  {bench_name + ' B&H':>12}"]
    out += [f"{a:<{w}}  {b:>24}  {c:>12}" for a, b, c in rows]
    if res.halted_on is not None:
        out.append(f"\n!! Kill switch triggered on {res.halted_on.date()} — strategy went to cash and stopped.")
    return "\n".join(out)
