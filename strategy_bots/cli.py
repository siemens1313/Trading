"""Command line: python -m strategy_bots <command>.

Daily bot: list, ingest, backtest, signal, paper, headlines. Intraday bot: execute, route, scalp,
make-market. Nothing here places real orders; the paper account is a JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

from . import backtest, text
from .data import DataError, Panel, ingest_robinhood, load_panel, load_sessions
from .daily import STRATEGIES
from .daily.base import DailyStrategy
from .intraday import INTRADAY, execution, market_making, scalping, sor
from .paper import PaperAccount

DAILY_DIR = Path("data/daily")
STATE_DIR = Path("state/strategy_bots")
USER_AGENT = "strategy-bots/0.1"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:  # DataError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strategy-bots", description="Two paper-trading bots for fifteen stock-market strategies."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="the strategies each bot runs").set_defaults(func=_list)

    ingest = sub.add_parser("ingest", help="save bars from saved Robinhood get_equity_historicals results as CSV")
    ingest.add_argument("files", nargs="+", type=Path)
    ingest.add_argument("--data", type=Path, default=Path("data"), help="writes DATA/daily and DATA/intraday (default: data)")
    ingest.set_defaults(func=_ingest)

    bt = sub.add_parser("backtest", help="walk-forward backtest of daily strategies against equal weight and SPY")
    _daily_args(bt, many=True)
    bt.add_argument("--start", help="first day to score, YYYY-MM-DD (default: after the longest warm-up)")
    bt.add_argument("--end", help="last day to score, YYYY-MM-DD")
    bt.add_argument("--json", type=Path, help="also write the metrics to this file")
    bt.set_defaults(func=_backtest)

    sig = sub.add_parser("signal", help="the target weights one daily strategy wants at the next open")
    _daily_args(sig, many=False)
    sig.set_defaults(func=_signal)

    paper = sub.add_parser("paper", help="advance a paper account: fill pending targets at each open, set new ones at the close")
    _daily_args(paper, many=False)
    paper.add_argument("--capital", type=float, default=10_000.0, help="starting cash for a new account (default: 10000)")
    paper.add_argument("--state", type=Path, help="account file (default: state/strategy_bots/paper_STRATEGY.json)")
    paper.set_defaults(func=_paper)

    news = sub.add_parser("headlines", help="append headlines from an RSS or Atom feed, for the sentiment strategy")
    news.add_argument("--feed", required=True, help="feed URL with {symbol} where the ticker goes")
    news.add_argument("--symbols", type=_symbols, required=True, help="comma-separated tickers")
    news.add_argument("--out", type=Path, default=Path("inputs/headlines.csv"), help="(default: inputs/headlines.csv)")
    news.set_defaults(func=_headlines)

    ex = sub.add_parser("execute", help="simulate TWAP, VWAP and implementation-shortfall execution on intraday bars")
    ex.add_argument("--bars", type=Path, required=True, help="intraday CSV, such as data/intraday/SPY_5minute.csv")
    ex.add_argument("--qty", type=int, required=True, help="shares in the parent order")
    ex.add_argument("--side", choices=("buy", "sell"), default="buy")
    ex.add_argument("--algo", choices=("twap", "vwap", "is", "all"), default="all")
    ex.add_argument("--day", help="session to trade, YYYY-MM-DD (default: the last); VWAP learns its volume profile from the sessions before it")
    ex.add_argument("--start", default="09:30", help="HH:MM New York time (default: 09:30)")
    ex.add_argument("--minutes", type=int, default=60, help="length of the execution window (default: 60)")
    ex.add_argument("--urgency", type=float, default=2.0, help="implementation-shortfall urgency; 0 is TWAP (default: 2)")
    ex.add_argument("--half-spread-bps", type=float, default=1.0, help="(default: 1)")
    ex.add_argument("--impact-bps", type=float, default=10.0, help="impact at 100%% of a bar's volume (default: 10)")
    ex.set_defaults(func=_execute)

    rt = sub.add_parser("route", help="simulate static against adaptive smart order routing")
    rt.add_argument("--rounds", type=int, default=200, help="orders to route (default: 200)")
    rt.add_argument("--qty", type=int, default=800, help="shares per order (default: 800)")
    rt.add_argument("--fade-bps", type=float, default=1.0, help="how far quotes move away on each re-route (default: 1)")
    rt.add_argument("--passes", type=int, default=10, help="routing passes per order (default: 10)")
    rt.add_argument("--seed", type=int, default=0)
    rt.set_defaults(func=_route)

    sc = sub.add_parser("scalp", help="simulate the scalper on intraday bars")
    sc.add_argument("--bars", type=Path, required=True, help="intraday CSV")
    sc.add_argument("--equity", type=float, default=10_000.0, help="account equity; under $25,000 the pattern-day-trader limit applies")
    sc.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override a scalper parameter")
    sc.set_defaults(func=_scalp)

    mm = sub.add_parser("make-market", help="simulate the market maker on intraday bars")
    mm.add_argument("--bars", type=Path, required=True, help="intraday CSV")
    mm.add_argument("--reference", type=Path, help="bars of a correlated stock for the stat-arb skew (with --param ref_beta=...)")
    mm.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override a market-maker parameter")
    mm.set_defaults(func=_make_market)
    return parser


def _daily_args(p: argparse.ArgumentParser, many: bool) -> None:
    if many:
        p.add_argument("--strategy", action="append", choices=list(STRATEGIES), help="repeat for several (default: all ten)")
    else:
        p.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    p.add_argument("--data", type=Path, default=DAILY_DIR, help="folder of SYMBOL.csv daily bars (default: data/daily)")
    p.add_argument("--symbols", type=_symbols, help="comma-separated subset of those files (default: all)")
    p.add_argument(
        "--param", action="append", default=[], metavar="[STRATEGY.]KEY=VALUE",
        help="override a strategy parameter; VALUE is read as JSON when it parses, else as text",
    )
    p.add_argument("--cost-bps", type=float, default=5.0, help="trading cost per unit of turnover, in bps (default: 5)")


def _symbols(text_: str) -> list[str]:
    out = [s.strip().upper() for s in text_.split(",") if s.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected comma-separated symbols")
    return out


def _value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _key_values(items: list[str]) -> list[tuple[str, object]]:
    out = []
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--param {item!r}: expected KEY=VALUE")
        out.append((key, _value(raw)))
    return out


def _strategy_params(items: list[str], names: list[str]) -> dict[str, dict]:
    """--param strategy.key=value, or key=value when only one strategy is selected."""
    out: dict[str, dict] = {n: {} for n in names}
    for key, value in _key_values(items):
        name, dot, field = key.partition(".")
        if not dot:
            if len(names) != 1:
                raise ValueError(f"--param {key}: say which strategy, as STRATEGY.{key}=...")
            name, field = names[0], key
        if name not in out:
            raise ValueError(f"--param {key}: {name} is not one of the selected strategies")
        out[name][field] = value
    return out


def _strategies(args) -> list[DailyStrategy]:
    if args.strategy is None:
        names = list(STRATEGIES)
    elif isinstance(args.strategy, str):
        names = [args.strategy]
    else:
        names = list(dict.fromkeys(args.strategy))
    params = _strategy_params(args.param, names)
    return [STRATEGIES[n](**params[n]) for n in names]


def _list(args) -> int:
    print("Daily bot: decides at each close what to hold from the next open (backtest, signal, paper --strategy NAME)")
    for name, cls in STRATEGIES.items():
        print(f"  #{cls.number:<3}{name:14}{cls.title}{'' if cls.long_only else '  [long/short]'}")
    print("Intraday bot: works inside the trading day on intraday bars, simulation only")
    for name, (number, title, command) in sorted(INTRADAY.items(), key=lambda kv: kv[1][0]):
        print(f"  #{number:<3}{name:14}{title}  ->  {command}")
    return 0


def _ingest(args) -> int:
    for path, note in ingest_robinhood(args.files, args.data).items():
        print(f"wrote {path} ({note})")
    return 0


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], left: int = 1) -> str:
    """Columns padded to fit; the first `left` columns are left-aligned, the rest right-aligned."""
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells):
        return "  ".join(str(c).ljust(w) if i < left else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    return "\n".join([line(headers), *(line(r) for r in rows)])


def _metric_cells(m: dict) -> list[str]:
    return [
        _pct(m["cagr"]), _pct(m["vol"]), _num(m["sharpe"]), _pct(m["max_drawdown"]),
        f"{_num(m['turnover'], 1)}x", _pct(m["exposure"], 0),
    ]


def _backtest(args) -> int:
    panel = load_panel(args.data, args.symbols)
    strategies = _strategies(args)
    if args.start:
        start = np.datetime64(args.start, "D")
        why = "as asked"
    else:
        slowest = max(strategies, key=lambda s: s.warmup())
        if slowest.warmup() >= len(panel) - 2:
            raise DataError(f"{len(panel)} days of data, but {slowest.name} needs {slowest.warmup()} days of history to trade")
        start = panel.dates[slowest.warmup() + 1]  # the day the first possible position is held
        why = f"after {slowest.name}'s {slowest.warmup()}-day warm-up, so every strategy is scored on the same days"
    end = np.datetime64(args.end, "D") if args.end else None
    rows, records, notes = [], [], []
    for s in strategies:
        r = backtest.run(panel, s.weights(panel), args.cost_bps, s.name).between(start, end)
        m = r.metrics()
        rows.append([f"#{s.number}", s.name, *_metric_cells(m)])
        records.append({"strategy": s.name, "number": s.number, "title": s.title, **m})
        if getattr(s, "note", None):
            notes.append(f"{s.name}: {s.note}")
        if s.warmup() + 1 < len(panel) and start < panel.dates[s.warmup() + 1]:
            notes.append(f"{s.name}: still warming up until {panel.dates[s.warmup() + 1]}; its early days are scored as cash")
    benchmarks = [backtest.equal_weight(panel)]
    if "SPY" in panel.symbols and len(panel.symbols) > 1:
        benchmarks.append(backtest.equal_weight(panel.select(["SPY"]), "buy & hold SPY"))
    for b in benchmarks:
        m = b.between(start, end).metrics()
        rows.append(["", b.name, *_metric_cells(m)])
        records.append({"strategy": b.name, "number": None, "title": b.name, **m})
    scored = benchmarks[0].between(start, end).dates
    if len(scored) < 2:
        raise DataError("fewer than two days to score between --start and --end")
    years = len(scored) / backtest.TRADING_DAYS
    print(
        f"Daily bot backtest: {len(panel.symbols)} symbols, scored {scored[0]} to {scored[-1]} ({years:.1f} years), "
        f"{args.cost_bps:g} bps per unit of turnover"
    )
    print(f"Scoring starts {why}. The benchmarks pay no costs.\n")
    print(_table(["#", "strategy", "CAGR", "vol", "Sharpe", "max DD", "turnover/yr", "exposure"], rows, left=2))
    for note in notes:
        print(f"\nnote - {note}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        meta = {"symbols": panel.symbols, "start": str(scored[0]), "end": str(scored[-1]), "cost_bps": args.cost_bps}
        clean = [{k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in rec.items()} for rec in records]
        args.json.write_text(json.dumps({**meta, "results": clean}, indent=1))
    return 0


def _targets(panel: Panel, row: np.ndarray) -> dict[str, float]:
    return {s: round(float(w), 6) for s, w in zip(panel.symbols, row) if abs(w) > 1e-9}


def _print_targets(targets: dict[str, float], indent: str = "  ") -> None:
    if not targets:
        print(f"{indent}no positions: all cash")
        return
    for s, w in sorted(targets.items(), key=lambda kv: -abs(kv[1])):
        print(f"{indent}{s:6} {w * 100:+6.1f}%")
    gross, net = sum(abs(w) for w in targets.values()), sum(targets.values())
    print(f"{indent}{'cash':6} {(1 - net) * 100:+6.1f}%" + (f"   (gross {gross * 100:.0f}%, net {net * 100:.0f}%)" if gross > net + 1e-9 else ""))


def _signal(args) -> int:
    panel = load_panel(args.data, args.symbols)
    (s,) = _strategies(args)
    weights = s.weights(panel)
    print(f"#{s.number} {s.title}")
    print(f"Targets decided at the {panel.dates[-1]} close, to trade at the next open:")
    _print_targets(_targets(panel, weights[-1]))
    for line in s.describe(panel):
        print(f"  - {line}")
    if len(panel) <= s.warmup():
        print(f"  - only {len(panel)} days of data; it needs {s.warmup()} before it trades")
    return 0


def _paper(args) -> int:
    panel = load_panel(args.data, args.symbols)
    (s,) = _strategies(args)
    path = args.state or STATE_DIR / f"paper_{s.name}.json"
    account = PaperAccount.open(path, args.capital)
    weights = s.weights(panel)
    dates = [str(d) for d in panel.dates]
    if account.pending and account.pending["decided"] > dates[-1]:
        raise DataError(f"{path} has targets from {account.pending['decided']}, after the last bar ({dates[-1]}): is the data stale?")
    first = len(panel) - 1
    if account.pending:
        first = next((t for t, d in enumerate(dates) if d > account.pending["decided"]), len(panel) - 1)
    fills = []
    for t in range(first, len(panel)):
        if account.pending and account.pending["decided"] < dates[t]:
            opens = {sym: float(panel.open[t, j]) for j, sym in enumerate(panel.symbols)}
            fills += account.rebalance(account.pending["targets"], opens, dates[t], args.cost_bps)
        account.pending = {"decided": dates[t], "targets": _targets(panel, weights[t])}
    closes = {sym: float(panel.close[-1, j]) for j, sym in enumerate(panel.symbols)}
    value = account.value(closes)
    account.save(path)
    print(f"Paper account for #{s.number} {s.title} ({path})")
    for f in fills:
        side = "bought" if f["shares"] > 0 else "sold"
        print(f"  {f['date']} open: {side} {abs(f['shares']):g} {f['symbol']} at {f['price']:.2f} (cost ${f['fee']:.2f})")
    if not fills:
        print("  no fills this run")
    print(f"Value at the {dates[-1]} close: ${value:,.2f}, of which cash ${account.cash:,.2f}")
    for sym, shares in sorted(account.positions.items()):
        worth = shares * closes[sym]
        print(f"  {sym:6} {shares:>12.4f} shares  ${worth:>12,.2f}  {worth / value * 100:5.1f}%")
    print(f"Targets for the next open (decided at the {dates[-1]} close):")
    _print_targets(account.pending["targets"])
    return 0


def _headlines(args) -> int:
    for symbol in args.symbols:
        url = args.feed.replace("{symbol}", symbol)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=20) as response:
            items = text.parse_feed(response.read())
        added = text.append_headlines(args.out, symbol, items)
        print(f"{symbol}: {len(items)} headlines in the feed, {added} new, saved to {args.out}")
    return 0


def _execute(args) -> int:
    sessions = load_sessions(args.bars)
    if not sessions:
        raise DataError(f"{args.bars}: no regular-session bars")
    if args.day:
        day = date.fromisoformat(args.day)
        index = next((i for i, s in enumerate(sessions) if s.day == day), None)
        if index is None:
            raise DataError(f"no session on {args.day} in {args.bars}")
    else:
        index = len(sessions) - 1
    session, past = sessions[index], sessions[:index]
    hour, _, minute = args.start.partition(":")
    at = int(hour) * 60 + int(minute or 0)
    hits = np.flatnonzero(session.minutes == at)
    if not len(hits):
        raise DataError(f"no bar starts at {args.start} on {session.day}")
    start = int(hits[0])
    step = int(np.median(np.diff(session.minutes))) if len(session) > 1 else 5
    bars = max(1, args.minutes // step)
    if start + bars > len(session):
        raise DataError(f"a {args.minutes}-minute window from {args.start} runs past the close")
    reports, skipped = [], []
    for algo in ("twap", "vwap", "is") if args.algo == "all" else (args.algo,):
        if algo == "twap":
            schedule = execution.twap(args.qty, bars)
        elif algo == "vwap":
            if not past:
                if args.algo == "all":
                    skipped.append("vwap: no earlier session to learn a volume profile from")
                    continue
                raise DataError("VWAP needs an earlier session to learn its volume profile from")
            schedule = execution.vwap(args.qty, execution.volume_profile(past, len(session))[start : start + bars])
        else:
            schedule = execution.implementation_shortfall(args.qty, bars, args.urgency)
        name = f"is (urgency {args.urgency:g})" if algo == "is" else algo
        reports.append(execution.simulate(session, schedule, start, args.side, name, args.half_spread_bps, args.impact_bps))
    print(f"{args.side.title()} {args.qty:,} shares on {session.day}, {args.start} for {bars * step} minutes ({bars} bars), from {args.bars}")
    rows = [
        [r.algo, f"{r.avg_price:.4f}", f"{r.arrival:.4f}", f"{r.interval_vwap:.4f}", f"{r.cost_vs_arrival_bps:+.2f}",
         f"{r.cost_vs_vwap_bps:+.2f}", _pct(r.max_participation)]
        for r in reports
    ]
    print(_table(["algo", "avg price", "arrival", "window VWAP", "vs arrival bps", "vs VWAP bps", "max part."], rows))
    print("Positive cost = worse than the benchmark. Fills are simulated: typical price, half-spread and square-root impact.")
    for note in skipped:
        print(f"note - {note}")
    if len(reports) == 1:
        print("Child orders:")
        for minute_, shares, price in reports[0].children:
            print(f"  {minute_ // 60:02d}:{minute_ % 60:02d}  {shares:>8,}  @ {price:.4f}")
    return 0


def _route(args) -> int:
    runs = [sor.simulate(args.rounds, args.qty, args.seed, adaptive, args.fade_bps, args.passes) for adaptive in (False, True)]
    print(f"{args.rounds} buy orders of {args.qty} shares across NYSE, NASDAQ, IEX and PHANTOM (best quote, but fills 20%)")
    rows = [
        ["adaptive" if r["adaptive"] else "static", _pct(r["fill_ratio"]), f"{r['avg_cost_bps_vs_mid']:.2f}",
         _pct(r["flow_share"]["PHANTOM"]), ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["learned_fill_rate"].items()))]
        for r in runs
    ]
    print(_table(["router", "filled", "cost vs arrival mid bps", "sent to PHANTOM", "learned fill rates"], rows))
    print("A simulation: retail brokers route your orders for you, so this can't be run live from a retail account.")
    return 0


def _scalp(args) -> int:
    sessions = load_sessions(args.bars)
    scalper = scalping.Scalper(**dict(_key_values(args.param)))
    result = scalper.run(sessions, args.equity)
    m = result.metrics()
    print(f"Scalper on {args.bars}: {len(sessions)} sessions, equity ${args.equity:,.0f}")
    if not m["trades"]:
        print(f"  no trades; {m['blocked_by_pdt']} signals refused by the pattern-day-trader limit")
        return 0
    print(
        f"  {m['trades']} trades, {_pct(m['win_rate'], 0)} winners, average {m['avg_bps']:+.1f} bps, total {m['total_bps']:+.1f} bps "
        f"after a {scalper.p['half_spread_bps']:g} bps half-spread each way"
    )
    if m["blocked_by_pdt"]:
        print(f"  {m['blocked_by_pdt']} more signals refused: under $25,000 only 3 day trades are allowed in 5 business days")
    for t in result.trades[-5:]:
        print(
            f"  {t.day} {t.entry_minute // 60:02d}:{t.entry_minute % 60:02d}-{t.exit_minute // 60:02d}:{t.exit_minute % 60:02d} "
            f"{t.entry:.2f} -> {t.exit:.2f}  {t.return_bps:+.1f} bps ({t.reason})"
        )
    return 0


def _make_market(args) -> int:
    sessions = load_sessions(args.bars)
    references = {s.day: s for s in load_sessions(args.reference)} if args.reference else {}
    maker = market_making.MarketMaker(**dict(_key_values(args.param)))
    rows, total = [], 0.0
    for s in sessions:
        r = maker.run(s, references.get(s.day))
        total += r.pnl
        rows.append([str(s.day), f"{r.pnl:+.2f}", str(r.fills), str(r.max_inventory), f"{r.spread_captured:+.2f}"])
    print(f"Market maker on {args.bars}, {maker.p['lot']}-share lots, at most {maker.p['max_lots']} lots either way")
    print(_table(["session", "P&L $", "fills", "max lots", "round trips $"], rows))
    print(f"Total ${total:+.2f} over {len(sessions)} sessions. Fills assume the next bar trading through a quote;")
    print("real queues, faster traders and fees make this optimistic. Simulation only.")
    return 0
