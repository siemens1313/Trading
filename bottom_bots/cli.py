"""Command line: python -m bottom_bots {run,list,backfill}."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

from . import config, manual, report, sources
from .bots import SENSOR_BOTS, AggregatorBot, run_sensors
from .history import History
from .sources import cboe
from .sources.http import get
from .store import SeriesStore

DEFAULT_MANUAL = Path("inputs/manual.toml")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            print(report.render_catalog())
            return 0
        if args.command == "backfill":
            return _backfill(args)
        return _run(args)
    except (config.ConfigError, manual.ManualInputError, tomllib.TOMLDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bottom-bots", description="Score the five-signal capitulation checklist.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fetch data, run all six bots, print the checklist")
    run.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="run date, YYYY-MM-DD")
    run.add_argument(
        "--manual",
        type=Path,
        action="append",
        help=f"manual inputs TOML, repeatable (default: {DEFAULT_MANUAL} if it exists)",
    )
    run.add_argument("--config", type=Path, help="TOML overriding bottom_bots/thresholds.toml")
    run.add_argument("--state-dir", type=Path, default=Path("state"))
    run.add_argument("--offline", action="store_true", help="skip live fetches; use the cache and manual inputs")
    run.add_argument(
        "--bot",
        action="append",
        choices=[bot.key for bot in SENSOR_BOTS],
        help="run only these sensor bots (no score; nothing saved)",
    )
    run.add_argument("--json", action="store_true")
    run.add_argument("--no-save", action="store_true", help="don't write the cache, history or report")

    sub.add_parser("list", help="describe the six bots and their data")

    backfill = sub.add_parser("backfill", help="fetch past Cboe put/call ratios for the one-year percentile")
    backfill.add_argument("--days", type=int, default=370)
    backfill.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    backfill.add_argument("--state-dir", type=Path, default=Path("state"))
    return parser


def _run(args: argparse.Namespace) -> int:
    cfg = config.load(args.config)
    cache = args.state_dir / "series.json"
    store = SeriesStore.load(cache)
    data_warnings = [] if args.offline else sources.refresh(store, cfg, args.as_of)

    manual_paths = args.manual or ([DEFAULT_MANUAL] if DEFAULT_MANUAL.exists() else [])
    for path in manual_paths:
        manual.load_into(store, path)
    if not manual_paths:
        data_warnings.append(f"no manual inputs ({DEFAULT_MANUAL} not found): manual-only items will be UNKNOWN")

    if args.bot:
        results = run_sensors(cfg, store, args.as_of, set(args.bot))
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        else:
            print(report.render_sensors(results, args.as_of, data_warnings), end="")
        return 0

    history = History.load(args.state_dir / "history.json")
    judge = AggregatorBot(cfg)
    verdict = judge.evaluate(run_sensors(cfg, store, args.as_of), history.before(args.as_of), args.as_of)
    markdown = report.render_markdown(verdict, data_warnings)
    print(report.render_json(verdict, data_warnings) + "\n" if args.json else markdown, end="")

    if not args.no_save:
        store.save(cache)
        history.put(judge.record(verdict))
        history.save(args.state_dir / "history.json")
        out = args.state_dir / "reports" / f"{args.as_of.isoformat()}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
    return 0


def _backfill(args: argparse.Namespace) -> int:
    cache = args.state_dir / "series.json"
    store = SeriesStore.load(cache)
    points, misses = cboe.backfill_put_call(args.as_of, args.days, get)
    for metric, day, value, source in points:
        store.add(metric, day, value, source)
    store.save(cache)
    print(f"added {len(points)} put/call values; {misses} weekday sessions unavailable")
    return 0 if points else 1
