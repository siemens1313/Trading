"""strategy_bots command line, end to end on synthetic CSV files."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from strategy_bots.cli import main
from strategy_bots.data import synthetic_panel, synthetic_sessions


def save_panel(panel, folder: Path, drop_last: int = 0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n = len(panel) - drop_last
    for j, symbol in enumerate(panel.symbols):
        with open(folder / f"{symbol}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for t in range(n):
                w.writerow([panel.dates[t], panel.open[t, j], panel.high[t, j], panel.low[t, j], panel.close[t, j], int(panel.volume[t, j])])


def save_sessions(sessions, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "open", "high", "low", "close", "volume"])
        for s in sessions:
            for k, minute in enumerate(s.minutes):
                w.writerow([f"{s.day} {minute // 60:02d}:{minute % 60:02d}", s.open[k], s.high[k], s.low[k], s.close[k], int(s.volume[k])])


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.panel = synthetic_panel(symbols=("AAA", "BBB", "CCC", "SPY"), days=320)
        save_panel(cls.panel, cls.dir / "daily")
        save_panel(cls.panel, cls.dir / "daily_old", drop_last=4)
        cls.bars = cls.dir / "XYZ_5minute.csv"
        save_sessions(synthetic_sessions(days=6), cls.bars)

    def test_list_names_all_fifteen(self):
        code, out, _ = run("list")
        self.assertEqual(code, 0)
        for number in range(1, 16):
            self.assertIn(f"#{number} ", out)

    def test_backtest_against_the_benchmarks(self):
        report = self.dir / "out" / "bt.json"
        code, out, _ = run("backtest", "--strategy", "momentum", "--strategy", "seasonality", "--data", str(self.dir / "daily"), "--json", str(report))
        self.assertEqual(code, 0, out)
        self.assertIn("after momentum's 200-day warm-up", out)
        saved = json.loads(report.read_text())
        self.assertEqual([r["strategy"] for r in saved["results"]], ["momentum", "seasonality", "equal weight, all symbols", "buy & hold SPY"])
        self.assertEqual(saved["start"], str(self.panel.dates[201]))  # the first day held after the warm-up decision

    def test_backtest_with_a_parameter_and_dates(self):
        code, out, _ = run(
            "backtest", "--strategy", "momentum", "--data", str(self.dir / "daily"), "--param", "lookback=63",
            "--start", str(self.panel.dates[150]), "--end", str(self.panel.dates[250]),
        )
        self.assertEqual(code, 0, out)
        self.assertIn(f"scored {self.panel.dates[150]} to {self.panel.dates[250]}", out)

    def test_bad_parameters_are_errors(self):
        daily = str(self.dir / "daily")
        code, _, err = run("backtest", "--strategy", "momentum", "--strategy", "pairs", "--data", daily, "--param", "lookback=63")
        self.assertEqual(code, 2)
        self.assertIn("say which strategy", err)
        code, _, err = run("backtest", "--strategy", "momentum", "--data", daily, "--param", "momentum.nope=1")
        self.assertEqual(code, 2)
        self.assertIn("unknown parameter", err)
        code, _, err = run("backtest", "--strategy", "rl", "--data", daily)
        self.assertEqual(code, 2)
        self.assertIn("needs 760 days", err)
        code, _, _ = run("signal", "--strategy", "momentum", "--data", str(self.dir / "nowhere"))
        self.assertEqual(code, 2)

    def test_signal(self):
        code, out, _ = run("signal", "--strategy", "adaptive", "--data", str(self.dir / "daily"))
        self.assertEqual(code, 0)
        self.assertIn(f"Targets decided at the {self.panel.dates[-1]} close", out)
        self.assertIn("trend-following:", out)

    def test_paper_account_catches_up_then_waits(self):
        state = self.dir / "state" / "acct.json"
        common = ["--strategy", "seasonality", "--param", 'symbols=["AAA"]', "--state", str(state)]
        code, out, _ = run("paper", "--data", str(self.dir / "daily_old"), *common)
        self.assertEqual(code, 0, out)
        self.assertIn("no fills this run", out)
        self.assertEqual(json.loads(state.read_text())["pending"]["decided"], str(self.panel.dates[-5]))

        code, out, _ = run("paper", "--data", str(self.dir / "daily"), *common)
        self.assertEqual(code, 0, out)
        account = json.loads(state.read_text())
        self.assertEqual(account["pending"]["decided"], str(self.panel.dates[-1]))
        self.assertEqual(len(account["history"]), 4)  # one rebalance at each of the four new opens

        code, out, _ = run("paper", "--data", str(self.dir / "daily"), *common)
        self.assertIn("no fills this run", out)
        self.assertEqual(len(json.loads(state.read_text())["history"]), 4)

        code, _, err = run("paper", "--data", str(self.dir / "daily_old"), *common)
        self.assertEqual(code, 2)
        self.assertIn("stale", err)

    def test_ingest(self):
        doc = {"data": {"results": [{"symbol": "QQQ", "interval": "day", "bars": [
            {"begins_at": "2026-01-05T00:00:00Z", "open_price": "1", "high_price": "2", "low_price": "0.5", "close_price": "1.5", "volume": 10, "interpolated": False}
        ]}]}}
        src = self.dir / "rh.json"
        src.write_text(json.dumps(doc))
        code, out, _ = run("ingest", str(src), "--data", str(self.dir / "ingested"))
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "ingested" / "daily" / "QQQ.csv").exists())

    def test_execute(self):
        code, out, _ = run("execute", "--bars", str(self.bars), "--qty", "5000", "--minutes", "30")
        self.assertEqual(code, 0, out)
        for algo in ("twap", "vwap", "is (urgency 2)"):
            self.assertIn(algo, out)
        code, out, _ = run("execute", "--bars", str(self.bars), "--qty", "600", "--algo", "twap", "--start", "10:00", "--minutes", "15", "--side", "sell")
        self.assertIn("Child orders:", out)
        self.assertEqual(out.count("@ "), 3)
        code, _, err = run("execute", "--bars", str(self.bars), "--qty", "100", "--algo", "vwap", "--day", "2026-01-05")
        self.assertEqual(code, 2)
        self.assertIn("earlier session", err)

    def test_route_scalp_and_make_market(self):
        code, out, _ = run("route", "--rounds", "40")
        self.assertEqual(code, 0)
        self.assertIn("adaptive", out)
        code, out, _ = run("scalp", "--bars", str(self.bars), "--equity", "50000")
        self.assertEqual(code, 0)
        self.assertIn("Scalper on", out)
        code, out, _ = run("make-market", "--bars", str(self.bars), "--reference", str(self.bars), "--param", "ref_beta=1.0")
        self.assertEqual(code, 0)
        self.assertIn("Total $", out)
        code, _, err = run("scalp", "--bars", str(self.bars), "--param", "nope")
        self.assertEqual(code, 2)

    def test_headlines_from_a_feed(self):
        feeds = self.dir / "feeds"
        feeds.mkdir(exist_ok=True)
        (feeds / "AAA.xml").write_text(
            "<rss><channel><item><title>AAA beats estimates</title><pubDate>Mon, 05 Jan 2026 15:00:00 GMT</pubDate></item></channel></rss>"
        )
        out_file = self.dir / "inputs" / "headlines.csv"
        args = ("headlines", "--feed", feeds.as_uri() + "/{symbol}.xml", "--symbols", "AAA", "--out", str(out_file))
        code, out, _ = run(*args)
        self.assertEqual(code, 0, out)
        self.assertIn("1 new", out)
        code, out, _ = run(*args)
        self.assertIn("0 new", out)
        self.assertEqual(out_file.read_text().splitlines()[1], "2026-01-05,AAA,AAA beats estimates")


if __name__ == "__main__":
    unittest.main()
