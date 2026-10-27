import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
from pathlib import Path

from bottom_bots.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "inputs" / "manual.example.toml"


def run_cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = main(list(argv))
    return code, out.getvalue()


def rows(metric: str, points: list[tuple[date, float]]) -> str:
    return "".join(f"[[{metric}]]\ndate = {d.isoformat()}\nvalue = {v}\n\n" for d, v in points)


def capitulation_inputs() -> str:
    """Two sessions (10/22, 10/23) on which all five checklist items are met."""
    d22, d23 = date(2026, 10, 22), date(2026, 10, 23)
    fridays = [d23 - timedelta(weeks=8 - i) for i in range(9)]  # 8/28 … 10/23
    closes, day = [], d23
    while len(closes) < 15:
        if day.weekday() < 5:
            closes.append(day)
        day -= timedelta(days=1)
    closes.reverse()
    bottom = len(closes) - 1 - 8
    return "".join(
        [
            rows("vix", [(d22, 35.0), (d23, 34.0)]),
            rows("pc_equity", [(d22, 0.92), (d23, 0.88)]),
            rows("gex", [(date(2026, 10, 20), -1.0), (date(2026, 10, 21), -0.5), (d22, 0.5), (d23, 0.8)]),
            rows("si_pct_float", [(date(2026, 10, 15), 15.0)]),
            rows("close.SOXX", [(d, 200.0 + abs(bottom - i)) for i, d in enumerate(closes)]),
            rows("etf_flows_weekly", list(zip(fridays, [-100, -100, -100, -100, 50, 50, 50, 50, 50]))),
            rows("aaii_bearish", list(zip(fridays[4:], [40, 52, 58, 54, 49]))),
        ]
    )


class CliTest(unittest.TestCase):
    def test_article_example_reproduces_its_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_cli(
                "run", "--offline", "--manual", str(EXAMPLE), "--as-of", "2026-10-23",
                "--state-dir", tmp, "--no-save", "--json",
            )
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(code, 0)
        data = json.loads(out)
        statuses = {r["key"]: r["status"] for r in data["results"]}
        self.assertEqual(
            statuses,
            {
                "volatility": "NOT_MET",
                "put_call": "NOT_MET",
                "gamma": "UNKNOWN",  # the article's only GEX figure is from 10/16: stale by 10/23
                "short_interest": "UNKNOWN",
                "flows_sentiment": "NOT_MET",
            },
        )
        self.assertEqual((data["score"], data["regime"], data["plan"]["target"]), (0, "NO SIGNAL", 0.0))
        self.assertTrue(any("Fear & Greed 21" in e for e in data["extremes"]))

    def test_two_days_of_capitulation_deploy_a_tranche(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "inputs.toml").write_text(capitulation_inputs())
            (tmp / "config.toml").write_text("[plan]\nreserve_cash = 10000.0\n")
            common = ["--offline", "--manual", str(tmp / "inputs.toml"), "--config", str(tmp / "config.toml"),
                      "--state-dir", str(tmp / "state")]

            code, first = run_cli("run", "--as-of", "2026-10-22", "--json", *common)
            self.assertEqual(code, 0)
            first = json.loads(first)
            self.assertEqual((first["score"], first["plan"]["target"], first["plan"]["runs_needed"]), (5, 0.0, 1))

            code, second = run_cli("run", "--as-of", "2026-10-23", "--json", *common)
            second = json.loads(second)
            self.assertEqual((second["score"], second["plan"]["target"]), (5, 0.75))
            self.assertEqual([o["notional"] for o in second["plan"]["orders"]], [3750.0, 3750.0])

            # Re-running a day replaces it rather than stacking another tranche.
            _, again = run_cli("run", "--as-of", "2026-10-23", "--json", *common)
            self.assertEqual(json.loads(again)["plan"]["increment"], 0.75)

            history = json.loads((tmp / "state" / "history.json").read_text())["runs"]
            self.assertEqual([r["as_of"] for r in history], ["2026-10-22", "2026-10-23"])
            report = (tmp / "state" / "reports" / "2026-10-23.md").read_text()
            self.assertIn("CAPITULATION-LIKE", report)
            self.assertIn("proposed: BUY QQQ $3,750.00", report)

    def test_single_bot_run(self):
        code, out = run_cli("run", "--offline", "--manual", str(EXAMPLE), "--as-of", "2026-10-23", "--no-save", "--bot", "put_call")
        self.assertEqual(code, 0)
        self.assertIn("Put/Call bot — NOT MET", out)
        self.assertNotIn("VIX term-structure", out)

    def test_bad_manual_input_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text("[[short_sale_volume]]\ndate = 2026-10-22\nvalue = 0.5\n")
            code, _ = run_cli("run", "--offline", "--manual", str(bad), "--no-save", "--state-dir", tmp)
        self.assertEqual(code, 2)

    def test_list(self):
        code, out = run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("Six bots", out)


if __name__ == "__main__":
    unittest.main()
