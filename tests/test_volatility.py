import unittest
from datetime import date, timedelta

from bottom_bots.bots.volatility import VolatilityBot
from bottom_bots.models import Status
from bottom_bots.store import SeriesStore
from tests.helpers import AS_OF, cfg, make_store


def run(store):
    return VolatilityBot(cfg(), store, AS_OF).evaluate()


class VolatilityBotTest(unittest.TestCase):
    def test_vix_above_30_is_met_even_without_vix3m(self):
        self.assertIs(run(make_store({"vix": 31.2})).status, Status.MET)

    def test_inverted_term_structure_is_met(self):
        result = run(make_store({"vix": 25.0, "vix3m": 24.0}))
        self.assertIs(result.status, Status.MET)
        self.assertIn("inverted", result.summary)

    def test_article_reading_is_not_met(self):
        result = run(make_store({"vix": 19.45, "vix3m": 20.48, "vxn": 28.30}))
        self.assertIs(result.status, Status.NOT_MET)
        ratio = next(r for r in result.readings if r.label == "VIX/VIX3M")
        self.assertAlmostEqual(ratio.value, 0.95, places=2)
        self.assertTrue(any(f.startswith("VXN − VIX = 8.") for f in result.flags))
        self.assertTrue(any("calm VIX" in n for n in result.notes))

    def test_calm_vix_without_vix3m_is_unknown(self):
        self.assertIs(run(make_store({"vix": 18.0})).status, Status.UNKNOWN)

    def test_stale_vix_is_unknown_with_a_warning(self):
        result = run(make_store({"vix": 35.0}, end=AS_OF - timedelta(days=10)))
        self.assertIs(result.status, Status.UNKNOWN)
        self.assertTrue(any("stale" in w for w in result.warnings))

    def test_ratio_is_taken_on_a_common_day(self):
        store = SeriesStore()
        store.add("vix", date(2026, 10, 22), 22.0, "test")
        store.add("vix", AS_OF, 23.0, "test")
        store.add("vix3m", date(2026, 10, 22), 22.5, "test")
        result = run(store)
        ratio = next(r for r in result.readings if r.label == "VIX/VIX3M")
        self.assertEqual(ratio.day, date(2026, 10, 22))
        self.assertAlmostEqual(ratio.value, 22.0 / 22.5)
        self.assertIs(result.status, Status.NOT_MET)

    def test_extreme_vix_cites_the_cboe_forward_returns(self):
        result = run(make_store({"vix": 42.0, "vix3m": 33.0}))
        self.assertTrue(any("extreme zone" in f for f in result.flags))
        self.assertTrue(any("+33%" in f for f in result.flags))

    def test_vxn_falling_while_semis_make_new_lows_warns(self):
        store = make_store(
            {
                "vix": 24.0,
                "vix3m": 23.0,
                "vxn": [36, 35, 34, 33, 32, 31],
                "close:SOXX": [300 - i for i in range(21)],
            }
        )
        self.assertTrue(any("not a bottom confirmation" in w for w in run(store).warnings))

    def test_no_divergence_warning_when_semis_hold_above_their_low(self):
        store = make_store(
            {
                "vix": 24.0,
                "vix3m": 23.0,
                "vxn": [36, 35, 34, 33, 32, 31],
                "close:SOXX": [300 - i for i in range(20)] + [295],
            }
        )
        self.assertEqual(run(store).warnings, [])


if __name__ == "__main__":
    unittest.main()
