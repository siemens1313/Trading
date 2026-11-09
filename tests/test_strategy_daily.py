"""strategy_bots daily bot: every strategy's contract (no look-ahead, bounds, warm-up) and each one's logic."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strategy_bots.daily import STRATEGIES
from strategy_bots.daily.adaptive import REVERSION, TREND
from strategy_bots.daily.base import cap_gross, first_fit, slots
from strategy_bots.daily.genetic import GENES, GeneticAlgorithm, _Indicators
from strategy_bots.daily.patterns import detect
from strategy_bots.daily.rl import DeepRL
from strategy_bots.data import Panel, synthetic_panel
from strategy_bots.indicators import zigzag

# Smaller training jobs so the whole suite runs in seconds; the logic is the same.
FAST = {
    "pairs": {"pairs": [["AAA", "BBB"]], "formation": 120},
    "seasonality": {"symbols": ["AAA"]},
    "ml": {"retrain_every": 100, "train_days": 300, "min_train_days": 200},
    "rl": {"retrain_every": 150, "train_days": 300, "min_train_days": 250, "episodes": 8},
    "genetic": {"population": 8, "generations": 3, "evolve_every": 100, "train_days": 300, "min_train_days": 200},
}


def panel_from_closes(closes: np.ndarray, symbols: list[str] | None = None, start: str = "2024-01-02") -> Panel:
    c = np.asarray(closes, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    symbols = symbols or [f"S{i}" for i in range(c.shape[1])]
    dates = np.busday_offset(np.datetime64(start, "D"), np.arange(len(c)), roll="forward")
    open_ = np.vstack([c[:1], c[:-1]])
    return Panel(symbols, dates, open_, np.maximum(open_, c) * 1.002, np.minimum(open_, c) * 0.998, c, np.full(c.shape, 1e6))


def write_headlines(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "headline"])
        w.writerows(rows)


class EveryStrategyTests(unittest.TestCase):
    """The contract every daily strategy keeps, checked on the same synthetic market."""

    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel(days=700)
        path = Path(tempfile.mkdtemp()) / "headlines.csv"
        days = cls.panel.dates[::9]
        write_headlines(path, [(str(d), "AAA" if i % 2 else "BBB", "Beats estimates, record profit" if i % 3 else "Misses, lawsuit") for i, d in enumerate(days)])
        cls.params = {**FAST, "sentiment": {"headlines": str(path)}}
        cls.weights = {name: cls.make(name).weights(cls.panel) for name in STRATEGIES}

    @classmethod
    def make(cls, name):
        return STRATEGIES[name](**cls.params.get(name, {}))

    def test_the_table_is_complete(self):
        self.assertEqual(sorted(cls.number for cls in STRATEGIES.values()), [1, 5, 8, 9, 10, 11, 12, 13, 14, 15])

    def test_shapes_bounds_and_sides(self):
        for name, w in self.weights.items():
            with self.subTest(name):
                self.assertEqual(w.shape, self.panel.close.shape)
                self.assertTrue(np.isfinite(w).all())
                self.assertLessEqual(np.abs(w).sum(axis=1).max(), 1 + 1e-9)
                if STRATEGIES[name].long_only:
                    self.assertGreaterEqual(w.min(), 0.0)

    def test_no_look_ahead(self):
        """Cutting off the future never changes a decision already made."""
        for name, w in self.weights.items():
            for n in (self.make(name).warmup() + 50, len(self.panel) - 30):
                with self.subTest(name, n=n):
                    np.testing.assert_array_equal(self.make(name).weights(self.panel.head(n)), w[:n])

    def test_nothing_trades_during_warmup(self):
        for name, w in self.weights.items():
            with self.subTest(name):
                self.assertFalse(w[: self.make(name).warmup()].any())

    def test_every_strategy_trades_here(self):
        for name, w in self.weights.items():
            with self.subTest(name):
                self.assertTrue(w.any(), f"{name} never traded")

    def test_unknown_parameters_are_refused(self):
        for cls in STRATEGIES.values():
            with self.subTest(cls.name), self.assertRaises(ValueError):
                cls(no_such_parameter=1)

    def test_describe_runs_after_weights(self):
        for name in STRATEGIES:
            s = self.make(name)
            s.weights(self.panel)
            with self.subTest(name):
                self.assertIsInstance(s.describe(self.panel), list)


class HelperTests(unittest.TestCase):
    def test_slots_keep_the_best_and_stick_with_holdings(self):
        active = np.array([True, True, True, False])
        scores = np.array([1.0, 3.0, 2.0, 9.0])
        np.testing.assert_array_equal(slots(active, 2, scores), [0, 0.5, 0.5, 0])
        np.testing.assert_array_equal(slots(active, 2, scores, held=np.array([True, False, False, True])), [0.5, 0.5, 0, 0])
        np.testing.assert_array_equal(slots(active, 5), [0.2, 0.2, 0.2, 0])

    def test_cap_gross(self):
        np.testing.assert_allclose(cap_gross(np.array([[0.8, -0.8], [0.2, 0.1]])), [[0.5, -0.5], [0.2, 0.1]])

    def test_first_fit(self):
        self.assertEqual(first_fit(100, 250, lambda t: (max(0, t - 300), t - 1)), 300)
        with self.assertRaises(ValueError):
            first_fit(100, 500, lambda t: (max(0, t - 300), t - 1))


class PairsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        T = 420
        b = np.log(50) + np.cumsum(rng.normal(0, 0.01, T))
        spread = np.zeros(T)
        for t in range(1, T):
            spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 0.004)
        spread[330:336] += 0.06  # A runs away from B for a week, then snaps back
        self.panel = panel_from_closes(np.exp(np.column_stack([b + 0.3 + spread, b])), ["AAA", "BBB"])

    def test_short_the_rich_leg_long_the_cheap_one(self):
        s = STRATEGIES["pairs"](pairs=[["AAA", "BBB"]])
        w = s.weights(self.panel)
        self.assertLess(w[330, 0], 0)
        self.assertGreater(w[330, 1], 0)
        self.assertAlmostEqual(np.abs(w[330]).sum(), 1.0)
        self.assertFalse(w[345:].any(), "it should be flat again once the spread has reverted")
        self.assertIsNotNone(s.models["AAA/BBB"])

    def test_unrelated_stocks_rarely_trade(self):
        rng = np.random.default_rng(5)
        walks = 50 * np.exp(np.cumsum(rng.normal(0, 0.015, (600, 2)), axis=0))
        w = STRATEGIES["pairs"](pairs=[["AAA", "BBB"]]).weights(panel_from_closes(walks, ["AAA", "BBB"]))
        self.assertLess((np.abs(w).sum(axis=1) > 0).mean(), 0.15)

    def test_missing_symbols_mean_no_trades(self):
        self.assertFalse(STRATEGIES["pairs"](pairs=[["XXX", "YYY"]]).weights(self.panel).any())


class MomentumTests(unittest.TestCase):
    def test_holds_the_leader_never_the_laggard(self):
        rng = np.random.default_rng(6)
        t = np.arange(400)[:, None]
        noise = np.cumsum(rng.normal(0, 0.004, (400, 3)), axis=0)
        closes = 100 * np.exp(np.array([0.002, 0.0, -0.002]) * t + noise)
        w = STRATEGIES["momentum"]().weights(panel_from_closes(closes, ["UP", "FLAT", "DOWN"]))
        self.assertGreater(w[-1, 0], 0)
        self.assertFalse(w[:, 2].any())


class SeasonalityTests(unittest.TestCase):
    def test_holds_exactly_the_turn_of_the_month(self):
        panel = synthetic_panel(days=300)
        w = STRATEGIES["seasonality"](symbols=["AAA"], month_filter=False).weights(panel)
        for t in range(len(panel)):
            hold = np.busday_offset(panel.dates[t], 1, roll="forward")
            month = hold.astype("datetime64[M]")
            first = np.busday_count(month.astype("datetime64[D]"), hold) < 3
            last = np.busday_count(hold, (month + 1).astype("datetime64[D]")) == 1
            self.assertEqual(w[t, 0] > 0, bool(first or last), str(panel.dates[t]))
        self.assertFalse(w[:, 1:].any())


class VolatilityBreakoutTests(unittest.TestCase):
    def test_buys_the_breakout_after_a_squeeze_and_stops_out(self):
        rng = np.random.default_rng(7)
        quiet, breakout, climb, crash = np.zeros(30), [0.04, 0.01, 0.01], np.full(20, 0.002), [-0.2]
        r = np.concatenate([rng.normal(0, 0.015, 170), quiet, breakout, climb, crash, np.full(5, 0.001)])
        w = STRATEGIES["vol_breakout"]().weights(panel_from_closes(100 * np.exp(np.cumsum(r))))
        self.assertFalse(w[170:200].any())  # nothing breaks out while it's quiet
        self.assertTrue((w[200:223, 0] > 0).all())  # long from the breakout close, through the climb
        self.assertFalse(w[223:].any())  # the crash hit the trailing stop


class MachineLearningTests(unittest.TestCase):
    def test_rebalances_only_on_the_horizon(self):
        s = STRATEGIES["ml"](**FAST["ml"])
        panel = synthetic_panel(days=500)
        w = s.weights(panel)
        h = s.p["horizon"]
        for t in range(1, len(panel)):
            if t % h:
                np.testing.assert_array_equal(w[t], w[t - 1])
        p = s.probabilities[np.isfinite(s.probabilities)]
        self.assertTrue(len(p) and ((p >= 0) & (p <= 1)).all())


class PatternTests(unittest.TestCase):
    P = STRATEGIES["patterns"].defaults

    def crossing(self, level: float) -> np.ndarray:
        return np.array([level - 0.01, level + 0.01])

    def test_double_bottom(self):
        pivots = [(0, np.log(100), -1), (10, np.log(110), 1), (20, np.log(100.5), -1)]
        found = detect(self.crossing(np.log(110)), pivots, 1, self.P)
        self.assertEqual(found[0], "double bottom")
        self.assertLess(found[1], np.log(100))  # stop under the lows
        self.assertGreater(found[2], np.log(110))  # target above the neckline

    def test_inverse_head_and_shoulders(self):
        pivots = [(0, np.log(100), -1), (5, np.log(110), 1), (10, np.log(90), -1), (15, np.log(110), 1), (20, np.log(100.5), -1)]
        self.assertEqual(detect(self.crossing(np.log(110)), pivots, 1, self.P)[0], "inverse head and shoulders")

    def test_ascending_triangle(self):
        pivots = [(0, np.log(110), 1), (5, np.log(100), -1), (10, np.log(110.5), 1), (15, np.log(104), -1)]
        self.assertEqual(detect(self.crossing(np.log(110.5)), pivots, 1, self.P)[0], "ascending triangle")

    def test_bull_flag(self):
        pole = np.log(np.linspace(100, 118, 15))
        flag = np.log(np.linspace(118, 115, 6))[1:]
        s = np.concatenate([np.full(30, np.log(100)), pole, flag, [np.log(118.5)]])
        self.assertEqual(detect(s, [], len(s) - 1, self.P)[0], "bull flag")

    def test_no_pattern_without_a_breakout(self):
        pivots = [(0, np.log(100), -1), (10, np.log(110), 1), (20, np.log(100.5), -1)]
        self.assertIsNone(detect(np.array([np.log(105), np.log(106)]), pivots, 1, self.P))

    def test_trades_a_double_bottom_end_to_end(self):
        path = np.concatenate([np.linspace(120, 100, 20), np.linspace(100, 112, 15), np.linspace(112, 100.5, 15), np.linspace(100.5, 116, 12)])
        s = STRATEGIES["patterns"]()
        w = s.weights(panel_from_closes(path))
        breakout = int(np.argmax(path[35:] > 112)) + 35
        self.assertGreater(w[breakout, 0], 0)
        self.assertFalse(w[:breakout].any())
        pivots = zigzag(np.log(path), self.P["swing"])
        self.assertTrue(all(c > i for i, _, _, c in pivots))


class SentimentTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        t = np.arange(120)[:, None]
        self.panel = panel_from_closes(100 * np.exp(0.001 * t + np.zeros((120, 2))), ["GOOD", "BAD"], start="2026-01-05")

    def test_buys_good_news_ignores_bad(self):
        path = self.dir / "h.csv"
        friday = str(self.panel.dates[60])
        saturday = str(np.datetime64(friday) + 1)
        write_headlines(path, [(saturday, "GOOD", "Beats estimates, raises guidance"), (saturday, "BAD", "Misses, lawsuit looms")])
        w = STRATEGIES["sentiment"](headlines=str(path)).weights(self.panel)
        self.assertFalse(w[60].any(), "Saturday's news can't be known at Friday's close")
        self.assertGreater(w[61, 0], 0)  # counted on Monday, the first trading day on or after it
        self.assertFalse(w[:, 1].any())

    def test_no_file_no_trades(self):
        s = STRATEGIES["sentiment"](headlines=str(self.dir / "missing.csv"))
        self.assertFalse(s.weights(self.panel).any())
        self.assertIn("not found", s.describe(self.panel)[0])


class ReinforcementLearningTests(unittest.TestCase):
    def test_agent_learns_which_action_pays(self):
        rng = np.random.default_rng(8)
        states = rng.normal(size=(300, 3, 8))
        for drift, wants_long in ((0.01, True), (-0.01, False)):
            agent = DeepRL(episodes=30, episode_len=60, step_days=1, seed=1)
            net = agent._train(states, np.full((300, 3), drift), seed=1)
            q = net.output(np.hstack([states[:, 0], np.zeros((300, 1))]))
            share_long = float((q[:, 1] > q[:, 0]).mean())
            with self.subTest(drift=drift):
                self.assertGreater(share_long if wants_long else 1 - share_long, 0.9)

    def test_decides_every_step_days(self):
        s = STRATEGIES["rl"](**FAST["rl"])
        w = s.weights(synthetic_panel(days=500))
        for t in range(1, len(w)):
            if t % s.p["step_days"]:
                np.testing.assert_array_equal(w[t], w[t - 1])


class AdaptiveTests(unittest.TestCase):
    def test_trend_mode_in_a_trend_reversion_in_chop(self):
        t = np.arange(300)
        trend = 100 * np.exp(0.003 * t)
        chop = 100 + 3 * np.sin(t / 2.0) + np.random.default_rng(1).normal(0, 0.3, 300)
        panel = panel_from_closes(np.column_stack([trend, chop]), ["TREND", "CHOP"])
        s = STRATEGIES["adaptive"]()
        w = s.weights(panel)
        self.assertEqual(s.modes[-1, 0], TREND)
        self.assertGreater(w[-1, 0], 0)
        self.assertNotIn(REVERSION, s.modes[:, 0])
        self.assertIn(REVERSION, s.modes[:, 1])
        self.assertEqual(s.describe(panel)[0], "trend-following: TREND")


class GeneticTests(unittest.TestCase):
    def test_elitism_never_loses_the_best(self):
        panel = synthetic_panel(days=400)
        ga = GeneticAlgorithm(population=10, generations=6)
        ind = _Indicators(panel.close)
        fwd = np.full(panel.close.shape, np.nan)
        fwd[:-2] = panel.open[2:] / panel.open[1:-1] - 1
        genome, fit, best = ga.evolve(ind, fwd, 50, 398, seed=3)
        self.assertTrue(all(b2 >= b1 for b1, b2 in zip(best, best[1:])))
        self.assertEqual(fit, best[-1])
        self.assertAlmostEqual(fit, ga.fitness(ind, fwd, 50, 398, genome))

    def test_repair_keeps_genomes_valid(self):
        rng = np.random.default_rng(9)
        for _ in range(200):
            fast, slow, rsi_len, floor = GeneticAlgorithm._repair(tuple(rng.uniform(-100, 300, 4)))
            self.assertLess(fast, slow)
            for value, (_, lo, hi) in zip((fast, slow, rsi_len, floor), GENES):
                self.assertTrue(lo <= value <= hi)


if __name__ == "__main__":
    unittest.main()
