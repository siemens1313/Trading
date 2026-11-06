from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from .backtest import build_panel, format_report, run_backtest
from .config import load_config
from .data import CSVProvider, RobinhoodProvider, YahooProvider, make_provider
from .engine import NY, EngineState, make_broker, market_open, run_cycle, signal_table

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def _print_cycle(rep) -> None:
    print(f"[{rep.ts}] mode={rep.mode}  equity=${rep.equity:,.2f}  cash=${rep.cash:,.2f}  "
          f"day {rep.day_pnl:+.2%}  drawdown {rep.drawdown:.1%}")
    d = rep.decision
    if d is not None:
        aw = ", ".join(f"{k} {v:.0%}" for k, v in d.alpha_weights.items() if v > 0) or "none (no proven edge)"
        print(f"forecast: combined IC {d.combined_ic:.3f} (after haircut) | alpha weights: {aw} | ex-ante vol {d.ex_ante_vol:.1%}")
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
    print("\nREAL-MONEY ORDERS (marketable limit, good for day):")
    for o in orders:
        print(f"  {o.side} {o.qty:.6f} {o.symbol} (~${o.notional:,.2f})")
    return input("Type 'send' to submit: ").strip() == "send"


def _bars(cfg):
    return CSVProvider(cfg.path(cfg.general.data_dir)).history(cfg.universe.symbols)


def cmd_backtest(cfg, a):
    bars = _bars(cfg)
    res = run_backtest(bars, cfg, start=a.start, end=a.end)
    print(format_report(res, cfg.universe.benchmark))
    out = cfg.path(cfg.general.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([res.equity, res.benchmark], axis=1).to_csv(out / "backtest_equity.csv", index_label="date")
    res.trades.to_csv(out / "backtest_trades.csv", index=False)
    res.weights.round(4).to_csv(out / "backtest_weights.csv", index_label="date")
    res.diag.to_csv(out / "backtest_diagnostics.csv", index_label="date")
    last = res.weights.iloc[-1]
    print(f"\nlast allocation ({res.weights.index[-1].date()}): "
          + ", ".join(f"{s} {w:.0%}" for s, w in last[last.abs() > 0.005].sort_values(ascending=False).items())
          + f", cash {1 - last.sum():.0%}")
    if a.report:
        from .report import write_tearsheet
        from .research import alpha_report, fmt_report
        rep, _ = alpha_report(bars, cfg, standalone=False)
        p = write_tearsheet(res, out / "tearsheet.html", cfg.universe.benchmark, fmt_report(rep))
        print(f"tearsheet: {p}")
    print(f"saved CSVs to {out}/")


def cmd_research(cfg, a):
    from .research import alpha_report, deflated, fmt_report, overlay_attribution
    bars = _bars(cfg)
    print(f"alpha research on {len(cfg.universe.symbols)} symbols (full sample, in-sample statistics)\n")
    rep, corr = alpha_report(bars, cfg, standalone=not a.fast)
    print(fmt_report(rep).to_string())
    print("\nscore correlations:")
    print(corr.round(2).to_string())
    res = run_backtest(bars, cfg)
    dsr = deflated(res, a.trials, a.trial_sharpe_std)
    print(f"\ncombined book: Sharpe {res.metrics['sharpe']:.2f}, 95% CI [{res.metrics['sharpe_ci'][0]:.2f}, "
          f"{res.metrics['sharpe_ci'][1]:.2f}], P(SR>0) {res.metrics['psr']:.0%}")
    print(f"deflated Sharpe probability (assuming {a.trials} configurations tried, cross-trial Sharpe sd "
          f"{a.trial_sharpe_std}): {dsr:.0%}")
    print("Read it as: the chance the edge is real after accounting for how many variants were tried.")
    if cfg.portfolio.long_only:
        at = overlay_attribution(bars, cfg, res)
        print(f"\nalpha overlay attribution (full book vs. same book on the beta prior alone):")
        print(f"  Sharpe {at['baseline_sharpe']:.2f} -> {at['full_sharpe']:.2f}   turnover {at['baseline_turnover']:.1f}x -> "
              f"{at['full_turnover']:.1f}x")
        print(f"  active return {at['active_return']:+.2%}/yr, tracking error {at['tracking_error']:.2%}, "
              f"IR {at['ir']:+.2f}, t-stat {at['t_stat']:+.2f}")
        verdict = ("ADDS VALUE net of costs" if at["t_stat"] > 2 else
                   "NOT PROVEN: it adds turnover without a statistically meaningful net gain")
        print(f"  verdict: alpha overlay {verdict}. (Set alphas.ic_haircut = 0 to run the beta book alone.)")


def cmd_run(cfg, a):
    rep = run_cycle(cfg, force=a.force, dry_run=a.dry_run, ignore_hours=a.ignore_hours,
                    confirm=(lambda o: True) if a.yes else _confirm)
    if a.verbose and rep.signals is not None:
        print(rep.signals.to_string(float_format=lambda x: f"{x:+.3f}"))
    _print_cycle(rep)


def cmd_loop(cfg, a):
    print(f"looping every {a.every} min (Ctrl-C to stop); rebalances at most once per {cfg.portfolio.rebalance} period")
    while True:
        try:
            _print_cycle(run_cycle(cfg, confirm=(lambda o: True) if a.yes else _confirm))
        except Exception as e:  # keep the loop alive, but make failures loud
            logging.exception("cycle failed: %s", e)
        time.sleep(a.every * 60)


def cmd_status(cfg, a):
    provider = make_provider(cfg)
    syms = cfg.universe.symbols
    quotes = provider.quotes(syms, cfg.costs.half_spread_bps)
    mids = {s: q.mid for s, q in quotes.items()}
    broker = make_broker(cfg)
    pos = broker.positions()
    cash = broker.available_cash(mids)
    eq = cash + sum(q * mids[s] for s, q in pos.items())
    st = EngineState.load(cfg.path(cfg.general.state_dir) / "engine_state.json")
    print(f"mode: {cfg.general.mode}   data: {cfg.general.data_provider}   market open: "
          f"{market_open(pd.Timestamp.now(tz=NY).to_pydatetime())}")
    if cfg.general.mode == "paper":
        print(f"paper account since {broker.created_at}: ${broker.starting_cash:,.2f} -> ${eq:,.2f} "
              f"({eq / broker.starting_cash - 1:+.2%})")
    print(f"equity ${eq:,.2f}  cash ${cash:,.2f}  peak ${st.peak_equity:,.2f}  last rebalance {st.last_rebalance_at or 'never'}")
    if st.halted:
        print(f"HALTED: {st.halted_reason}")
    w = pd.Series({s: q * mids[s] / eq for s, q in pos.items()}, dtype=float).reindex(syms, fill_value=0.0)
    today = pd.Timestamp(pd.Timestamp.now(tz=NY).date())
    bars = {s: df[df.index < today] for s, df in provider.history(syms).items()}
    panel = build_panel(bars, cfg)
    T = panel.C.index[-1]
    beta = float((w * panel.beta.loc[T].reindex(syms).fillna(0)).sum())
    print(f"exposure: gross {w.abs().sum():.0%}  net {w.sum():.0%}  beta {beta:.2f}  positions {int((w != 0).sum())}")
    print(f"\nalpha state as of {T.date()}:")
    print(pd.DataFrame({"realized IC": panel.fc.alpha_ic.loc[T], "weight": panel.fc.alpha_weights.loc[T]})
          .to_string(float_format=lambda x: f"{x:+.3f}"))
    print(f"combined IC used for sizing: {panel.fc.combined_ic.loc[T]:.3f}  signal persistence: {panel.fc.persistence.loc[T]:.2f}")
    print("\nbook:")
    print(signal_table(panel, T, None, w).to_string(float_format=lambda x: f"{x:+.3f}"))


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
    if not a.yes and input("Erase the paper account, journal and engine state? [y/N] ").lower() != "y":
        return
    d = cfg.path(cfg.general.state_dir)
    for f in ("paper_account.json", "paper_trades.csv", "engine_state.json", "paper_equity.csv", "journal.jsonl"):
        (d / f).unlink(missing_ok=True)
    print("paper account reset")


def cmd_update_data(cfg, a):
    src = {"yahoo": YahooProvider, "robinhood": RobinhoodProvider}[a.source]()
    bars = src.history(cfg.universe.symbols)
    CSVProvider(cfg.path(cfg.general.data_dir)).save(bars)
    for s, df in bars.items():
        print(f"{s}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="quantbot", description="Desk-style systematic trading bot (paper first)")
    ap.add_argument("-c", "--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="backtest the full pipeline on data/*.csv")
    b.add_argument("--start"); b.add_argument("--end")
    b.add_argument("--report", action="store_true", help="also write reports/tearsheet.html")
    rs = sub.add_parser("research", help="per-alpha IC, decay, persistence, standalone Sharpe, deflated Sharpe")
    rs.add_argument("--trials", type=int, default=15, help="how many configurations you have tried (be honest)")
    rs.add_argument("--trial-sharpe-std", type=float, default=0.15)
    rs.add_argument("--fast", action="store_true", help="skip standalone backtests")
    r = sub.add_parser("run", help="run one trading cycle (paper or live)")
    r.add_argument("--force", action="store_true", help="rebalance now even if already done this period")
    r.add_argument("--dry-run", action="store_true", help="show orders without sending them")
    r.add_argument("--ignore-hours", action="store_true", help="trade even if the market is closed")
    r.add_argument("--yes", action="store_true", help="skip the live-mode confirmation prompt")
    r.add_argument("-v", "--verbose", action="store_true", help="print the per-symbol forecast table")
    lp = sub.add_parser("loop", help="run cycles forever")
    lp.add_argument("--every", type=float, default=15, help="minutes between cycles")
    lp.add_argument("--yes", action="store_true")
    sub.add_parser("status", help="account, exposures, alpha state and the per-symbol book")
    rz = sub.add_parser("resume", help="clear the kill switch")
    rz.add_argument("--reset-peak", action="store_true")
    rp = sub.add_parser("reset-paper", help="erase the paper account")
    rp.add_argument("--yes", action="store_true")
    u = sub.add_parser("update-data", help="refresh data/*.csv")
    u.add_argument("--source", choices=["yahoo", "robinhood"], default="yahoo")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    {"backtest": cmd_backtest, "research": cmd_research, "run": cmd_run, "loop": cmd_loop, "status": cmd_status,
     "resume": cmd_resume, "reset-paper": cmd_reset_paper, "update-data": cmd_update_data}[a.cmd](cfg, a)
