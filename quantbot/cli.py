from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .backtest import format_report, run_backtest
from .config import load_config
from .data import CSVProvider, YahooProvider, RobinhoodProvider, align, make_provider
from .engine import EngineState, make_broker, market_open, run_cycle, NY
from .strategy import explain

pd.set_option("display.width", 120)


def _print_report(rep) -> None:
    print(f"[{rep.ts}] mode={rep.mode}  equity=${rep.equity:,.2f}  cash=${rep.cash:,.2f}  drawdown={rep.drawdown:.1%}")
    print(f"action: {rep.action}")
    for o in rep.orders:
        print(f"  order  {o.side:<4} {o.qty:>12.6f} {o.symbol:<5} @ ~{o.ref_price:,.2f}  (${o.notional:,.2f})")
    for o, why in rep.rejected:
        print(f"  REJECT {o.side:<4} {o.qty:>12.6f} {o.symbol:<5}: {why}")
    for f in rep.fills:
        print(f"  {f.status:<9} {f.side:<4} {f.qty:>12.6f} {f.symbol:<5} @ {f.price:,.4f} {f.note}")


def _confirm(orders) -> bool:
    if not sys.stdin.isatty():
        print("live mode needs an interactive confirmation (or --yes)")
        return False
    print("\nREAL-MONEY ORDERS:")
    for o in orders:
        print(f"  {o.side} {o.qty:.6f} {o.symbol} (~${o.notional:,.2f})")
    return input("Type 'send' to submit: ").strip() == "send"


def cmd_backtest(cfg, a):
    bars = CSVProvider(cfg.path(cfg.general.data_dir)).history(cfg.universe.symbols)
    res = run_backtest(bars, cfg, start=a.start, end=a.end)
    print(format_report(res, cfg.universe.benchmark))
    out = cfg.path(cfg.general.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([res.equity, res.benchmark], axis=1).to_csv(out / "backtest_equity.csv", index_label="date")
    res.trades.to_csv(out / "backtest_trades.csv", index=False)
    res.weights.round(4).to_csv(out / "backtest_weights.csv", index_label="date")
    print(f"\nlast allocation ({res.weights.index[-1].date()}):")
    last = res.weights.iloc[-1]
    print("  " + ", ".join(f"{s} {w:.0%}" for s, w in last[last > 0.001].items()) + f", cash {1 - last.sum():.0%}")
    print(f"saved equity curve, trades and weights to {out}/")


def cmd_run(cfg, a):
    rep = run_cycle(cfg, force=a.force, dry_run=a.dry_run, ignore_hours=a.ignore_hours,
                    confirm=(lambda o: True) if a.yes else _confirm)
    if rep.signals is not None and not rep.signals.empty and a.verbose:
        print(rep.signals.to_string(float_format=lambda x: f"{x:.4f}"))
    _print_report(rep)


def cmd_loop(cfg, a):
    print(f"looping every {a.every} min (Ctrl-C to stop); rebalances happen at most once per {cfg.strategy.rebalance} period")
    while True:
        try:
            rep = run_cycle(cfg, confirm=(lambda o: True) if a.yes else _confirm)
            _print_report(rep)
        except Exception as e:  # keep the loop alive, but make failures loud
            logging.exception("cycle failed: %s", e)
        time.sleep(a.every * 60)


def cmd_status(cfg, a):
    provider = make_provider(cfg)
    syms = cfg.universe.symbols
    prices = provider.latest_prices(syms)
    broker = make_broker(cfg)
    pos = broker.positions()
    cash = broker.available_cash(prices)
    eq = cash + sum(q * prices[s] for s, q in pos.items())
    st = EngineState.load(cfg.path(cfg.general.state_dir) / "engine_state.json")
    print(f"mode: {cfg.general.mode}   data: {cfg.general.data_provider}")
    if cfg.general.mode == "paper":
        start = broker.starting_cash
        print(f"paper account since {broker.created_at}: start ${start:,.2f} -> ${eq:,.2f} ({eq / start - 1:+.2%})")
    print(f"equity ${eq:,.2f}   cash ${cash:,.2f}   peak ${st.peak_equity:,.2f}   last rebalance {st.last_rebalance_at or 'never'}")
    if st.halted:
        print(f"HALTED: {st.halted_reason}")
    if pos:
        rows = [(s, q, prices[s], q * prices[s], q * prices[s] / eq) for s, q in sorted(pos.items())]
        print(pd.DataFrame(rows, columns=["symbol", "qty", "price", "value", "weight"]).to_string(index=False,
              float_format=lambda x: f"{x:,.4f}"))
    else:
        print("no positions")
    closes = align(provider.history(syms), "close")
    closes = closes[closes.index < pd.Timestamp(pd.Timestamp.now(tz=NY).date())]
    print("\ncurrent signals:")
    print(explain(closes, cfg.strategy).to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nmarket open now: {market_open(pd.Timestamp.now(tz=NY).to_pydatetime())}")


def cmd_resume(cfg, a):
    p = cfg.path(cfg.general.state_dir) / "engine_state.json"
    st = EngineState.load(p)
    st.halted, st.halted_reason = False, ""
    if a.reset_peak:
        st.peak_equity = 0.0
    st.save()
    print("kill switch cleared" + (" and peak reset" if a.reset_peak else ""))


def cmd_reset_paper(cfg, a):
    if cfg.general.mode != "paper":
        sys.exit("reset-paper only works in paper mode")
    if not a.yes and input("Erase the paper account and engine state? [y/N] ").lower() != "y":
        return
    d = cfg.path(cfg.general.state_dir)
    for f in ("paper_account.json", "paper_trades.csv", "engine_state.json", "paper_equity.csv"):
        (d / f).unlink(missing_ok=True)
    print("paper account reset")


def cmd_update_data(cfg, a):
    src = {"yahoo": YahooProvider, "robinhood": RobinhoodProvider}[a.source]()
    bars = src.history(cfg.universe.symbols)
    CSVProvider(cfg.path(cfg.general.data_dir)).save(bars)
    for s, df in bars.items():
        print(f"{s}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quantbot", description="Quant trading bot (paper first, Robinhood live optional)")
    ap.add_argument("-c", "--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="backtest on data/*.csv")
    b.add_argument("--start"); b.add_argument("--end")
    r = sub.add_parser("run", help="run one trading cycle (paper or live)")
    r.add_argument("--force", action="store_true", help="rebalance now even if already done this period")
    r.add_argument("--dry-run", action="store_true", help="show orders without sending them")
    r.add_argument("--ignore-hours", action="store_true", help="trade even if the market is closed")
    r.add_argument("--yes", action="store_true", help="skip the live-mode confirmation prompt")
    r.add_argument("-v", "--verbose", action="store_true", help="print the signal table")
    lp = sub.add_parser("loop", help="run cycles forever")
    lp.add_argument("--every", type=float, default=15, help="minutes between cycles")
    lp.add_argument("--yes", action="store_true")
    sub.add_parser("status", help="account, positions and current signals")
    rs = sub.add_parser("resume", help="clear the kill switch")
    rs.add_argument("--reset-peak", action="store_true")
    rp = sub.add_parser("reset-paper", help="erase the paper account")
    rp.add_argument("--yes", action="store_true")
    u = sub.add_parser("update-data", help="refresh data/*.csv")
    u.add_argument("--source", choices=["yahoo", "robinhood"], default="yahoo")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    {"backtest": cmd_backtest, "run": cmd_run, "loop": cmd_loop, "status": cmd_status, "resume": cmd_resume,
     "reset-paper": cmd_reset_paper, "update-data": cmd_update_data}[a.cmd](cfg, a)
