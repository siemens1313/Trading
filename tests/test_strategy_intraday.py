"""strategy_bots intraday bot: execution schedules and fills, smart order routing, scalping, market making."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np

from strategy_bots.data import SESSION_OPEN, Session, synthetic_sessions
from strategy_bots.intraday import INTRADAY, execution, market_making, scalping, sor


def session(closes, day=date(2026, 1, 5), opens=None, highs=None, lows=None, volume=10_000.0) -> Session:
    c = np.asarray(closes, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else np.concatenate([[c[0]], c[:-1]])
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, c)
    l = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, c)
    v = np.full(len(c), volume) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    return Session(day, SESSION_OPEN + 5 * np.arange(len(c)), o, h, l, c, v)


class TableTests(unittest.TestCase):
    def test_the_intraday_rows(self):
        self.assertEqual(sorted(number for number, _, _ in INTRADAY.values()), [2, 3, 4, 6, 7])


class ScheduleTests(unittest.TestCase):
    def test_allocate_sums_exactly(self):
        for qty in (0, 1, 7, 1000, 12345):
            out = execution.allocate(qty, np.array([0.3, 0.3, 0.4]))
            self.assertEqual(out.sum(), qty)
        np.testing.assert_array_equal(execution.allocate(10, np.array([1.0, 0.0, 1.0])), [5, 0, 5])
        self.assertEqual(execution.allocate(10, np.zeros(3)).sum(), 0)

    def test_twap_is_even_and_vwap_follows_volume(self):
        np.testing.assert_array_equal(execution.twap(12, 4), [3, 3, 3, 3])
        np.testing.assert_array_equal(execution.vwap(100, np.array([0.5, 0.25, 0.25])), [50, 25, 25])

    def test_implementation_shortfall_front_loads_with_urgency(self):
        calm = execution.implementation_shortfall(1200, 12, 0.0)
        np.testing.assert_array_equal(calm, execution.twap(1200, 12))
        eager = execution.implementation_shortfall(1200, 12, 3.0)
        self.assertEqual(eager.sum(), 1200)
        self.assertTrue(np.all(np.diff(eager) <= 0))
        self.assertGreater(eager[0], 3 * eager[-1])

    def test_volume_profile_uses_complete_sessions(self):
        days = synthetic_sessions(days=5)
        profile = execution.volume_profile(days)
        self.assertAlmostEqual(profile.sum(), 1.0)
        self.assertGreater(profile[0], profile[len(profile) // 2])  # busy open, quiet lunch
        short = session(np.ones(10))
        np.testing.assert_allclose(execution.volume_profile(days + [short], len(days[0])), profile)
        with self.assertRaises(ValueError):
            execution.volume_profile([short], 78)


class FillTests(unittest.TestCase):
    def test_costs_and_sides(self):
        s = session(np.full(12, 100.0))
        buy = execution.simulate(s, execution.twap(1200, 12), side="buy", half_spread_bps=1.0, impact_bps=10.0)
        sell = execution.simulate(s, execution.twap(1200, 12), side="sell", half_spread_bps=1.0, impact_bps=10.0)
        expected = 1.0 + 10.0 * np.sqrt(100 / 10_000)  # half-spread plus sqrt impact at 1% participation
        self.assertAlmostEqual(buy.cost_vs_arrival_bps, expected)
        self.assertAlmostEqual(sell.cost_vs_arrival_bps, expected)
        self.assertAlmostEqual(buy.max_participation, 0.01)
        self.assertEqual(sum(shares for _, shares, _ in buy.children), 1200)

    def test_bigger_orders_cost_more(self):
        s = session(np.full(12, 100.0))
        small = execution.simulate(s, execution.twap(1200, 12))
        large = execution.simulate(s, execution.twap(48_000, 12))
        self.assertGreater(large.cost_vs_arrival_bps, small.cost_vs_arrival_bps)

    def test_rejects_bad_requests(self):
        s = session(np.full(5, 100.0))
        with self.assertRaises(ValueError):
            execution.simulate(s, execution.twap(100, 6))
        with self.assertRaises(ValueError):
            execution.simulate(s, execution.twap(100, 5), side="short")


class RouterTests(unittest.TestCase):
    def test_effective_price_includes_fees(self):
        v = sor.Venue("X", 99.99, 100.01, 100, 100, fee_bps=3.0)
        self.assertAlmostEqual(v.effective("buy"), 100.01 * 1.0003)
        self.assertAlmostEqual(v.effective("sell"), 99.99 * 0.9997)

    def test_plan_takes_the_best_prices_up_to_displayed_size(self):
        venues = [sor.Venue("A", 99.9, 100.02, 300, 300), sor.Venue("B", 99.9, 100.01, 200, 200), sor.Venue("C", 99.9, 100.05, 900, 900)]
        plan = sor.SmartOrderRouter(adaptive=False).plan("buy", 600, venues)
        self.assertEqual([(v.name, n) for v, n in plan], [("B", 200), ("A", 300), ("C", 100)])

    def test_adaptive_ranking_learns_from_misses(self):
        router = sor.SmartOrderRouter(adaptive=True)
        for _ in range(10):
            router.record("FLAKY", 100, 10)
        self.assertAlmostEqual(router.fill_rate["FLAKY"], 0.1 + 0.9 * 0.8**10)
        flaky, solid = sor.Venue("FLAKY", 99.9, 100.005, 500, 500), sor.Venue("SOLID", 99.9, 100.01, 500, 500)
        self.assertEqual(router.plan("buy", 100, [flaky, solid])[0][0].name, "SOLID")
        self.assertEqual(sor.SmartOrderRouter(adaptive=False).plan("buy", 100, [flaky, solid])[0][0].name, "FLAKY")

    def test_adaptive_router_beats_static_in_simulation(self):
        static, adaptive = sor.simulate(adaptive=False), sor.simulate(adaptive=True)
        self.assertGreater(adaptive["fill_ratio"], static["fill_ratio"])
        self.assertLess(adaptive["avg_cost_bps_vs_mid"], static["avg_cost_bps_vs_mid"])
        self.assertLess(adaptive["flow_share"]["PHANTOM"], 0.05)
        self.assertGreater(static["flow_share"]["PHANTOM"], 0.3)


class ScalperTests(unittest.TestCase):
    def buying_pressure(self, day=date(2026, 1, 5), target=True) -> Session:
        """Three strong up bars, then (bar 4) a jump to the profit target or a drop to the stop."""
        closes = [100.0, 100.1, 100.2, 100.3, 100.3, 100.3, 100.3, 100.3]
        s = session(closes, day=day)
        if target:
            s.high[4] = 100.6
        else:
            s.low[4] = 99.9
        return s

    def test_target_and_stop(self):
        result = scalping.Scalper().run([self.buying_pressure()], equity=50_000)
        self.assertEqual([t.reason for t in result.trades], ["target"])
        self.assertAlmostEqual(result.trades[0].return_bps, ((1 + 15e-4) * (1 - 1e-4) - 1) * 1e4)
        result = scalping.Scalper().run([self.buying_pressure(target=False)], equity=50_000)
        self.assertEqual([t.reason for t in result.trades], ["stop"])

    def test_stop_wins_when_a_bar_touches_both(self):
        s = self.buying_pressure()
        s.low[4] = 99.9
        self.assertEqual(scalping.Scalper().run([s], equity=50_000).trades[0].reason, "stop")

    def test_pattern_day_trader_limit(self):
        days = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 12, 13)]
        sessions = [self.buying_pressure(day) for day in days]
        small = scalping.Scalper().run(sessions, equity=5_000)
        # Mon-Wed trade; Thu and Fri would make a 4th day trade in 5 business days; by Mon 12th the 5th has
        # rolled out of the window, and by Tue 13th the 6th has too.
        self.assertEqual([t.day.day for t in small.trades], [5, 6, 7, 12, 13])
        self.assertEqual(small.blocked_by_pdt, 2)
        for t in small.trades:
            recent = [u for u in small.trades if 0 <= np.busday_count(np.datetime64(u.day), np.datetime64(t.day)) < 5]
            self.assertLessEqual(len(recent), 3)
        self.assertEqual(len(scalping.Scalper().run(sessions, equity=30_000).trades), 7)


class MarketMakerTests(unittest.TestCase):
    def test_quotes_straddle_the_mid_and_lean_on_inventory(self):
        mm = market_making.MarketMaker()
        bid, ask = mm.quotes(100.0, sigma_bps=5.0, lots=0, bars_left=40)
        self.assertLess(bid, 100.0)
        self.assertGreater(ask, 100.0)
        long_bid, long_ask = mm.quotes(100.0, sigma_bps=5.0, lots=3, bars_left=40)
        self.assertLess(long_bid, bid)
        self.assertLess(long_ask, ask)

    def test_the_stat_arb_skew_never_crosses_the_mid(self):
        mm = market_making.MarketMaker()
        bid, ask = mm.quotes(100.0, sigma_bps=5.0, lots=0, bars_left=40, shift_bps=50.0)
        self.assertLessEqual(bid, 100.0 + 1e-9)
        self.assertGreater(ask, 100.0)

    def test_inventory_stays_inside_the_limit(self):
        mm = market_making.MarketMaker(max_lots=2)
        for s in synthetic_sessions(days=4):
            result = mm.run(s)
            self.assertLessEqual(result.max_inventory, 2)
            self.assertTrue(all(abs(x) <= 2 for x in result.inventory))
            self.assertGreater(result.fills, 0)

    def test_round_trip_in_a_flat_market_earns_the_spread(self):
        n = 60
        mid = np.full(n, 100.0)
        s = session(mid, highs=mid * 1.002, lows=mid * 0.998)  # every bar trades through both quotes
        result = market_making.MarketMaker(vol_bars=5).run(s)
        self.assertGreater(result.fills, 0)
        self.assertGreater(result.pnl, 0)

    def test_unknown_parameters_are_refused(self):
        with self.assertRaises(ValueError):
            market_making.MarketMaker(nope=1)
        with self.assertRaises(ValueError):
            scalping.Scalper(nope=1)


if __name__ == "__main__":
    unittest.main()
