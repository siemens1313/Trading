"""Daily event-driven backtest of the full desk pipeline.

Timing (no lookahead): at the close of day T, alpha scores use closes <= T; alpha ICs use only
returns realized by the open of T (see alphas.py); the risk model uses returns <= T. Orders are filled
at the OPEN of T+1 with half-spread + slippage + square-root impact. The live engine does the same:
it uses completed bars, then trades at the current quote.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alphas import Forecast, build_forecast
from .broker import SimulatedBroker
from .config import Config
from .costs import fill_cost_bps
from .data import align
from .execution import Order, plan_orders
from .portfolio import decide, rolling_beta
from .risk import RiskManager
from .riskmodel import ewma_vol
from .stats import summarize


def period_key(ts: pd.Timestamp, freq: str):
    if freq == "daily":
        return ts.date()
    iso = ts.isocalendar()
    return (iso[0], iso[1])


@dataclass
class Panel:
    O: pd.DataFrame
    C: pd.DataFrame
    rets: pd.DataFrame
    ewma: pd.DataFrame
    adv: pd.DataFrame   # 20-day average dollar volume
    beta: pd.DataFrame
    fc: Forecast


def build_panel(bars: dict[str, pd.DataFrame], cfg: Config) -> Panel:
    O, C, V = align(bars, "open"), align(bars, "close"), align(bars, "volume")
    idx = O.index.intersection(C.index).intersection(V.index)
    O, C, V = O.loc[idx], C.loc[idx], V.loc[idx]
    rets = C.pct_change()
    fc = build_forecast(C, O, cfg.alphas, cfg.universe.benchmark)
    beta = rolling_beta(rets, cfg.universe.benchmark, cfg.portfolio.beta_window)
    return Panel(O, C, rets, ewma_vol(rets, cfg.risk_model.vol_halflife), (C * V).rolling(20).mean(), beta, fc)


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    diag: pd.DataFrame          # per-day: gross, net, ex-ante vol, combined IC, turnover, cost
    alpha_weights: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict
    halted_on: pd.Timestamp | None
    panel: Panel


def run_backtest(bars: dict[str, pd.DataFrame], cfg: Config, start: str | None = None,
                 end: str | None = None, panel: Panel | None = None) -> BacktestResult:
    P = panel or build_panel(bars, cfg)
    dates = P.C.index if end is None else P.C.loc[:end].index
    ready = P.fc.alpha_ic.dropna(how="all").index
    if len(ready) == 0:
        raise ValueError("alphas never accumulated enough IC history; need more data")
    warm = max(ready[0], P.ewma.dropna(how="all").index[0], P.beta.dropna(how="all").index[0])
    start_ts = max(pd.Timestamp(start), warm) if start else warm
    # include the day before start so the first signal is formed at start's close
    port, ex, rk, cc = cfg.portfolio, cfg.execution, cfg.risk, cfg.costs
    broker = SimulatedBroker(rk.starting_cash, 0.0, 0.0, ex.fractional, allow_short=not port.long_only)
    risk = RiskManager(rk)
    pending: pd.Series | None = None
    peak, halted_on = 0.0, None
    eq, wts, diag, trades = {}, {}, {}, []
    last_key = None

    for i, t in enumerate(dates):
        cost_today, traded_today = 0.0, 0.0
        if pending is not None and halted_on is None:
            o_px = P.O.loc[t].to_dict()
            e_open = broker.equity(o_px)
            prev = dates[i - 1]
            for o in plan_orders(pending, broker.positions(), o_px, e_open, ex):
                bps = fill_cost_bps(o.notional, P.adv.at[prev, o.symbol], P.ewma.at[prev, o.symbol], cc)
                f = broker.submit(o, o_px[o.symbol], ts=str(t.date()), cost_bps=bps)
                if f.status == "filled":
                    c = f.qty * o_px[o.symbol] * bps / 1e4
                    cost_today += c
                    traded_today += f.qty * o_px[o.symbol]
                    trades.append({"date": t, "symbol": f.symbol, "side": f.side, "qty": f.qty,
                                   "price": f.price, "cost_bps": bps, "cost": c})
            pending = None
        if t < start_ts:
            continue

        c_px = P.C.loc[t].to_dict()
        e = broker.equity(c_px)
        peak = max(peak, e)
        if halted_on is None and risk.kill_switch(e, peak):
            for s, q in list(broker.positions().items()):
                side = "sell" if q > 0 else "buy"
                broker.submit(Order(s, side, abs(q), c_px[s]), c_px[s], ts=str(t.date()), cost_bps=cc.linear_bps)
            e, halted_on = broker.equity(c_px), t
        w_now = pd.Series({s: q * c_px[s] / e for s, q in broker.positions().items()}, dtype=float)
        w_now = w_now.reindex(P.C.columns, fill_value=0.0)
        eq[t], wts[t] = e, w_now
        row = {"gross": w_now.abs().sum(), "net": w_now.sum(), "turnover": traded_today / e, "cost": cost_today,
               "combined_ic": float(P.fc.combined_ic.loc[t]), "ex_ante_vol": np.nan}

        key = period_key(t, port.rebalance)
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        due = port.rebalance == "daily" or (nxt is not None and period_key(nxt, "weekly") != key) or last_key is None
        if halted_on is None and nxt is not None and due:
            d = decide(t, P.fc, P.rets, P.ewma, w_now, cfg, P.beta)
            if d is not None:
                pending, last_key = d.weights, key
                row["ex_ante_vol"] = d.ex_ante_vol
        diag[t] = row

    equity = pd.Series(eq, name="strategy")
    bench = P.C[cfg.universe.benchmark].loc[equity.index]
    benchmark = (bench / bench.iloc[0] * rk.starting_cash).rename(cfg.universe.benchmark)
    tr = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price", "cost_bps", "cost"])
    weights = pd.DataFrame(wts).T
    dg = pd.DataFrame(diag).T
    m = summarize(equity)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    m.update({
        "trades": len(tr), "turnover_per_year": float(dg["turnover"].sum() / years),
        "costs_total": float(tr["cost"].sum()), "cost_drag_per_year": float((dg["cost"] / equity.shift(1)).sum() / years),
        "avg_gross": float(dg["gross"].mean()), "avg_net": float(dg["net"].mean()),
        "beta": float(np.cov(equity.pct_change().dropna(), benchmark.pct_change().dropna())[0, 1]
                      / benchmark.pct_change().var()),
    })
    return BacktestResult(equity, benchmark, weights, tr, dg, P.fc.alpha_weights.loc[equity.index],
                          m, summarize(benchmark), halted_on, P)


def format_report(res: BacktestResult, bench_name: str) -> str:
    m, b = res.metrics, res.benchmark_metrics
    f = lambda x, s: format(x, s)  # noqa: E731
    rows = [
        ("Period", f"{m['start']} -> {m['end']}", ""),
        ("Total return", f(m["total_return"], "+.1%"), f(b["total_return"], "+.1%")),
        ("CAGR", f(m["cagr"], "+.2%"), f(b["cagr"], "+.2%")),
        ("Ann. volatility", f(m["ann_vol"], ".2%"), f(b["ann_vol"], ".2%")),
        ("Sharpe (rf=0)", f(m["sharpe"], ".2f"), f(b["sharpe"], ".2f")),
        ("  95% CI (block bootstrap)", f"[{m['sharpe_ci'][0]:.2f}, {m['sharpe_ci'][1]:.2f}]",
         f"[{b['sharpe_ci'][0]:.2f}, {b['sharpe_ci'][1]:.2f}]"),
        ("  P(Sharpe > 0)", f(m["psr"], ".0%"), f(b["psr"], ".0%")),
        ("Sortino", f(m["sortino"], ".2f"), f(b["sortino"], ".2f")),
        ("Max drawdown", f(m["max_drawdown"], ".1%"), f(b["max_drawdown"], ".1%")),
        ("Calmar", f(m["calmar"], ".2f"), f(b["calmar"], ".2f")),
        ("Beta to " + bench_name, f(m["beta"], ".2f"), "1.00"),
        ("Avg gross / net exposure", f"{m['avg_gross']:.0%} / {m['avg_net']:.0%}", "100% / 100%"),
        ("Turnover / year", f(m["turnover_per_year"], ".1f") + "x", ""),
        ("Cost drag / year", f(m["cost_drag_per_year"], ".2%"), ""),
        ("Trades", str(m["trades"]), ""),
    ]
    w = max(len(r[0]) for r in rows)
    out = [f"{'':<{w}}  {'Strategy':>24}  {bench_name + ' B&H':>14}"]
    out += [f"{a:<{w}}  {x:>24}  {y:>14}" for a, x, y in rows]
    if res.halted_on is not None:
        out.append(f"\n!! Kill switch triggered on {res.halted_on.date()}; book flattened and halted.")
    return "\n".join(out)
