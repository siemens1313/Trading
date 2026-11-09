"""strategy_bots shared core: data, indicators, backtest timing, paper account, models, headline text."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from strategy_bots import backtest, indicators, learn, text
from strategy_bots.data import (
    DataError,
    Panel,
    ingest_robinhood,
    load_daily_csv,
    load_panel,
    load_sessions,
    synthetic_panel,
)
from strategy_bots.daily import STRATEGIES
from strategy_bots.paper import PaperAccount, advance


def write_daily(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def rh_bar(begins_at: str, price: float, interpolated: bool = False) -> dict:
    return {
        "begins_at": begins_at, "open_price": str(price), "high_price": str(price + 1), "low_price": str(price - 1),
        "close_price": str(price + 0.5), "volume": 1000, "interpolated": interpolated,
    }


class DataTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_daily_csv_is_sorted_by_date(self):
        write_daily(self.dir / "AAA.csv", [("2026-01-06", 2, 2, 2, 2, 10), ("2026-01-05", 1, 1, 1, 1, 10)])
        loaded = load_daily_csv(self.dir / "AAA.csv")
        self.assertEqual([str(d) for d in loaded["dates"]], ["2026-01-05", "2026-01-06"])
        self.assertEqual(list(loaded["close"]), [1.0, 2.0])

    def test_panel_keeps_only_shared_dates(self):
        write_daily(self.dir / "AAA.csv", [(f"2026-01-0{d}", d, d, d, d, 1) for d in (5, 6, 7)])
        write_daily(self.dir / "BBB.csv", [(f"2026-01-0{d}", 10 * d, 10 * d, 10 * d, 10 * d, 1) for d in (6, 7, 8)])
        panel = load_panel(self.dir)
        self.assertEqual(panel.symbols, ["AAA", "BBB"])
        self.assertEqual([str(d) for d in panel.dates], ["2026-01-06", "2026-01-07"])
        np.testing.assert_array_equal(panel.close, [[6, 60], [7, 70]])
        self.assertEqual(len(panel.head(1)), 1)
        self.assertEqual(panel.select(["BBB"]).close.tolist(), [[60.0], [70.0]])

    def test_panel_errors(self):
        with self.assertRaises(DataError):
            load_panel(self.dir)
        write_daily(self.dir / "AAA.csv", [("2026-01-05", 1, 1, 1, 1, 1)])
        with self.assertRaises(DataError):
            load_panel(self.dir, ["AAA", "ZZZ"])

    def test_ingest_robinhood_daily_and_intraday(self):
        doc = {"data": {"results": [
            {"symbol": "AAA", "interval": "day", "bars": [rh_bar("2026-01-05T00:00:00Z", 10), rh_bar("2026-01-06T00:00:00Z", 11)]},
            {"symbol": "AAA", "interval": "5minute", "bars": [
                rh_bar("2026-01-05T14:30:00Z", 10),  # 09:30 New York
                rh_bar("2026-01-05T14:35:00Z", 10.2),
                rh_bar("2026-01-05T14:40:00Z", 10.4, interpolated=True),
                rh_bar("2026-01-05T21:05:00Z", 10.6),  # 16:05, after the close
            ]},
        ]}}
        src = self.dir / "rh.json"
        src.write_text(json.dumps(doc))
        written = ingest_robinhood([src], self.dir)
        self.assertEqual(sorted(Path(p).name for p in written), ["AAA.csv", "AAA_5minute.csv"])
        daily = load_daily_csv(self.dir / "daily" / "AAA.csv")
        self.assertEqual(list(daily["open"]), [10.0, 11.0])
        sessions = load_sessions(self.dir / "intraday" / "AAA_5minute.csv")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].day, date(2026, 1, 5))
        self.assertEqual(list(sessions[0].minutes), [570, 575])  # the interpolated and after-hours bars are gone

        # a second ingest merges: new values replace old ones, old dates stay
        doc["data"]["results"] = [{"symbol": "AAA", "interval": "day", "bars": [rh_bar("2026-01-06T00:00:00Z", 12), rh_bar("2026-01-07T00:00:00Z", 13)]}]
        src.write_text(json.dumps(doc))
        ingest_robinhood([src], self.dir)
        self.assertEqual(list(load_daily_csv(self.dir / "daily" / "AAA.csv")["open"]), [10.0, 12.0, 13.0])

    def test_ingest_rejects_bad_input(self):
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"data": {}}))
        with self.assertRaises(DataError):
            ingest_robinhood([bad], self.dir)
        bad.write_text(json.dumps({"data": {"results": [{"symbol": "A", "interval": "day", "bars": [rh_bar("2026-01-05T00:00:00Z", 0)]}]}}))
        with self.assertRaises(DataError):
            ingest_robinhood([bad], self.dir)

    def test_synthetic_panel_is_consistent(self):
        p = synthetic_panel(days=50)
        self.assertEqual(p.close.shape, (50, 4))
        self.assertTrue((p.high >= np.maximum(p.open, p.close)).all())
        self.assertTrue((p.low <= np.minimum(p.open, p.close)).all())


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(100, 5, (60, 2))

    def test_rolling_stats_match_naive(self):
        n = 7
        for t in (n - 1, 30, 59):
            np.testing.assert_allclose(indicators.sma(self.x, n)[t], self.x[t - n + 1 : t + 1].mean(axis=0))
            np.testing.assert_allclose(indicators.rolling_std(self.x, n)[t], self.x[t - n + 1 : t + 1].std(axis=0, ddof=1))
            np.testing.assert_allclose(indicators.rolling_max(self.x, n)[t], self.x[t - n + 1 : t + 1].max(axis=0))
        self.assertTrue(np.isnan(indicators.sma(self.x, n)[: n - 1]).all())

    def test_rsi_extremes_and_range(self):
        up = np.arange(1.0, 40.0)[:, None]
        self.assertAlmostEqual(indicators.rsi(up, 14)[-1, 0], 100.0)
        self.assertAlmostEqual(indicators.rsi(up[::-1].copy(), 14)[-1, 0], 0.0)
        r = indicators.rsi(self.x, 14)
        self.assertTrue(np.isnan(r[:14]).all())
        self.assertTrue(((r[15:] >= 0) & (r[15:] <= 100)).all())

    def test_efficiency_ratio_and_kama(self):
        line = np.linspace(100, 120, 40)[:, None]
        self.assertAlmostEqual(indicators.efficiency_ratio(line, 10)[-1, 0], 1.0)
        zigzag_path = np.tile([100.0, 101.0], 20)[:, None]
        self.assertLess(indicators.efficiency_ratio(zigzag_path, 10)[-1, 0], 0.15)
        k = indicators.kama(line, 10)
        self.assertTrue(np.all(np.diff(k[~np.isnan(k[:, 0]), 0]) > 0))  # follows a steady trend up

    def test_bollinger_and_atr(self):
        mid, upper, lower, width = indicators.bollinger(self.x, 20, 2.0)
        np.testing.assert_allclose((upper - mid)[19:], (mid - lower)[19:])
        np.testing.assert_allclose(width[19:], ((upper - lower) / mid)[19:])
        a = indicators.atr(self.x + 1, self.x - 1, self.x, 14)
        self.assertTrue((a[20:] >= 2 - 1e-9).all())  # the range alone is 2

    def test_zigzag_pivots_are_confirmed_later_and_causal(self):
        s = np.log(np.array([100, 110, 121, 115, 104, 100, 106, 118, 130, 117, 105], dtype=float))
        pivots = indicators.zigzag(s, 0.08)
        self.assertEqual([(i, kind) for i, _, kind, _ in pivots], [(0, -1), (2, 1), (5, -1), (8, 1)])
        self.assertEqual([c for *_, c in pivots], [1, 4, 7, 9])  # the day the move reached 8%
        for i, _, _, confirmed in pivots:
            self.assertGreater(confirmed, i)
        # pivots confirmed by day t don't change when the series is cut at t
        for t in range(len(s)):
            seen = [p for p in pivots if p[3] <= t]
            self.assertEqual([p for p in indicators.zigzag(s[: t + 1], 0.08) if p[3] <= t], seen)


class BacktestTests(unittest.TestCase):
    def panel(self, opens: list[float]) -> Panel:
        o = np.array(opens, dtype=float)[:, None]
        dates = np.busday_offset(np.datetime64("2026-01-05"), np.arange(len(o)), roll="forward")
        return Panel(["AAA"], dates, o, o, o, o, np.ones_like(o))

    def test_weights_trade_at_the_next_open(self):
        panel = self.panel([100, 100, 110, 121, 121])
        w = np.zeros((5, 1))
        w[1] = 1.0  # decided at the close of day 1: bought at the day-2 open, sold at the day-3 open
        r = backtest.run(panel, w, cost_bps=0.0)
        np.testing.assert_allclose(r.returns, [0.0, 0.10, 0.0])
        self.assertEqual([str(d) for d in r.dates], ["2026-01-06", "2026-01-07", "2026-01-08"])

    def test_costs_on_every_change(self):
        panel = self.panel([100] * 6)
        w = np.zeros((6, 1))
        w[1:3] = 1.0
        r = backtest.run(panel, w, cost_bps=10.0)
        np.testing.assert_allclose(r.turnover, [0, 1, 0, 1])
        np.testing.assert_allclose(r.returns, [0, -0.001, 0, -0.001])

    def test_metrics(self):
        m = backtest.summarize(np.full(252, 0.001), np.zeros(252), np.ones(252))
        self.assertAlmostEqual(m["total_return"], 1.001**252 - 1)
        self.assertAlmostEqual(m["cagr"], 1.001**252 - 1)
        self.assertEqual(m["max_drawdown"], 0.0)
        m = backtest.summarize(np.array([0.1, -0.5, 0.2]), np.zeros(3), np.ones(3))
        self.assertAlmostEqual(m["max_drawdown"], -0.5)
        self.assertTrue(math.isnan(backtest.summarize(np.array([0.1]), np.zeros(1), np.ones(1))["sharpe"]))

    def test_between_and_shape_check(self):
        panel = self.panel([100, 101, 102, 103, 104, 105])
        r = backtest.equal_weight(panel).between("2026-01-07", "2026-01-08")
        self.assertEqual([str(d) for d in r.dates], ["2026-01-07", "2026-01-08"])
        with self.assertRaises(ValueError):
            backtest.run(panel, np.zeros((5, 1)))


class PaperTests(unittest.TestCase):
    def test_rebalance_once_per_day_with_costs(self):
        acct = PaperAccount(cash=10_000.0)
        fills = acct.rebalance({"AAA": 0.5}, {"AAA": 100.0}, "2026-01-05", cost_bps=10)
        self.assertEqual(fills[0]["shares"], 50.0)
        self.assertAlmostEqual(acct.cash, 10_000 - 5_000 - 5.0)
        self.assertEqual(acct.rebalance({"AAA": 1.0}, {"AAA": 100.0}, "2026-01-05"), [])
        self.assertAlmostEqual(acct.value({"AAA": 110.0}), 4_995 + 5_500)
        fills = acct.rebalance({}, {"AAA": 110.0}, "2026-01-06", cost_bps=0)
        self.assertEqual(fills[0]["shares"], -50.0)
        self.assertEqual(acct.positions, {})

    def test_band_skips_drift_trades_but_not_exits(self):
        acct = PaperAccount(cash=0.0, positions={"AAA": 50.0, "BBB": 1.0})
        fills = acct.rebalance({"AAA": 0.99}, {"AAA": 100.0, "BBB": 50.0}, "2026-01-05", band=0.01)
        self.assertEqual([f["symbol"] for f in fills], ["BBB"])  # AAA is ~1% off target; BBB is an exit

    def test_shorts_and_missing_prices(self):
        acct = PaperAccount(cash=1_000.0)
        acct.rebalance({"AAA": -0.5}, {"AAA": 10.0}, "2026-01-05", cost_bps=0)
        self.assertEqual(acct.positions, {"AAA": -50.0})
        self.assertAlmostEqual(acct.value({"AAA": 12.0}), 1_500 - 600)
        with self.assertRaises(KeyError):
            acct.value({})

    def test_save_and_open_keep_pending_targets(self):
        path = Path(tempfile.mkdtemp()) / "acct.json"
        acct = PaperAccount.open(path, 5_000.0)
        acct.pending = {"decided": "2026-01-05", "targets": {"AAA": 0.2}}
        acct.save(path)
        again = PaperAccount.open(path, 1.0)
        self.assertEqual(again.cash, 5_000.0)
        self.assertEqual(again.pending, acct.pending)


class AdvanceTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel(days=320)
        self.strategy = STRATEGIES["momentum"]

    def test_a_replay_matches_daily_runs(self):
        since = str(self.panel.dates[290])
        daily = PaperAccount(cash=10_000.0, capital=10_000.0)
        for n in range(291, len(self.panel) + 1):
            head = self.panel.head(n)
            advance(daily, head, self.strategy().weights(head), since=since)
        replay = PaperAccount(cash=10_000.0, capital=10_000.0)
        fills = advance(replay, self.panel, self.strategy().weights(self.panel), since=since)
        self.assertTrue(fills)
        self.assertEqual(replay.positions, daily.positions)
        self.assertAlmostEqual(replay.cash, daily.cash, places=6)
        self.assertEqual(replay.marks, daily.marks)
        self.assertEqual(replay.pending, daily.pending)
        self.assertEqual((replay.opened, daily.opened), (since, since))

    def test_first_targets_wait_for_the_next_open(self):
        account = PaperAccount(cash=1_000.0, capital=1_000.0)
        weights = np.full(self.panel.close.shape, 0.25)
        self.assertEqual(advance(account, self.panel.head(100), weights[:100]), [])
        self.assertEqual(account.pending["decided"], str(self.panel.dates[99]))
        fills = advance(account, self.panel.head(101), weights[:101])
        self.assertEqual({f["date"] for f in fills}, {str(self.panel.dates[100])})
        self.assertAlmostEqual(fills[0]["price"], float(self.panel.open[100, 0]))

    def test_refuses_stale_data_and_a_start_after_the_data(self):
        account = PaperAccount(cash=1_000.0)
        weights = np.zeros(self.panel.close.shape)
        advance(account, self.panel, weights)
        with self.assertRaises(ValueError):
            advance(account, self.panel.head(300), weights[:300])
        with self.assertRaises(ValueError):
            advance(PaperAccount(cash=1.0), self.panel, weights, since="2099-01-01")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(3000, 5))
        logits = 2.5 * X[:, 0] - 2.0 * X[:, 1] * (X[:, 2] > 0)
        y = (rng.random(3000) < learn.sigmoid(logits)).astype(float)
        cls.train, cls.test = (X[:2000], y[:2000]), (X[2000:], y[2000:])

    def accuracy(self, model) -> float:
        X, y = self.test
        return float(((model.predict_proba(X) > 0.5) == y).mean())

    def test_models_learn_a_planted_signal(self):
        X, y = self.train
        self.assertGreater(self.accuracy(learn.RandomForest(seed=0).fit(X, y)), 0.72)
        self.assertGreater(self.accuracy(learn.MLP([5, 16, 1], seed=0).fit_binary(X, y)), 0.72)
        self.assertGreater(self.accuracy(learn.LinearSVM(seed=0).fit(X, y)), 0.7)

    def test_probabilities_are_probabilities(self):
        X, y = self.train
        for model in (learn.RandomForest(seed=0).fit(X, y), learn.LinearSVM(seed=0).fit(X, y)):
            p = model.predict_proba(self.test[0])
            self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_standardizer(self):
        X = np.random.default_rng(2).normal(5, 3, (500, 3))
        Z = learn.Standardizer().fit(X).transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-2)


class TextTests(unittest.TestCase):
    def test_score(self):
        self.assertEqual(text.score("Apple beats estimates, raises guidance"), 1.0)
        self.assertEqual(text.score("Intel misses; shares plunge"), -1.0)
        self.assertEqual(text.score("Company fails to beat estimates"), -1.0)
        self.assertEqual(text.score("Intel fails to beat; lawsuit looms"), -1.0)
        self.assertEqual(text.score("Shareholder meeting scheduled"), 0.0)
        self.assertEqual(text.score("Upgrade, but lawsuit"), 0.0)

    def test_parse_feeds_dated_in_new_york(self):
        rss = b"""<rss><channel>
            <item><title>Late news</title><pubDate>Tue, 06 Jan 2026 01:30:00 GMT</pubDate></item>
            <item><title>No date</title></item>
        </channel></rss>"""
        self.assertEqual(text.parse_feed(rss), [("2026-01-05", "Late news")])  # 8:30 pm the day before, in New York
        atom = b"""<feed xmlns="http://www.w3.org/2005/Atom">
            <entry><title>Atom item</title><updated>2026-01-06T15:00:00Z</updated></entry></feed>"""
        self.assertEqual(text.parse_feed(atom), [("2026-01-06", "Atom item")])

    def test_append_and_load_headlines(self):
        path = Path(tempfile.mkdtemp()) / "h.csv"
        self.assertEqual(text.append_headlines(path, "aaa", [("2026-01-05", "Beats")]), 1)
        self.assertEqual(text.append_headlines(path, "AAA", [("2026-01-05", "Beats"), ("2026-01-06", "Misses")]), 1)
        rows = text.load_headlines(path)
        self.assertEqual([(str(d), s, h) for d, s, h in rows], [("2026-01-05", "AAA", "Beats"), ("2026-01-06", "AAA", "Misses")])


if __name__ == "__main__":
    unittest.main()
