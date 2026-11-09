import unittest

from bottom_bots.bots.gamma import GammaBot
from bottom_bots.models import Status
from tests.helpers import AS_OF, cfg, make_store


def run(store):
    return GammaBot(cfg(), store, AS_OF).evaluate()


class GammaBotTest(unittest.TestCase):
    def test_negative_gamma_is_not_met(self):
        result = run(make_store({"gex": [-1.0, -2.0]}))
        self.assertIs(result.status, Status.NOT_MET)
        self.assertIn("amplified", result.summary)

    def test_flip_from_negative_is_met(self):
        self.assertIs(run(make_store({"gex": [-2.0, -1.0, -0.5, 0.4]})).status, Status.MET)

    def test_positive_throughout_is_not_a_flip(self):
        result = run(make_store({"gex": [1.0, 2.0, 3.0]}))
        self.assertIs(result.status, Status.NOT_MET)
        self.assertIn("no flip", result.summary)

    def test_a_single_positive_reading_is_unknown(self):
        self.assertIs(run(make_store({"gex": 0.5})).status, Status.UNKNOWN)

    def test_negative_reading_outside_the_lookback_does_not_count(self):
        self.assertIs(run(make_store({"gex": [-1.0] + [1.0] * 25})).status, Status.NOT_MET)

    def test_index_below_the_flip_level_is_not_met(self):
        result = run(make_store({"gex": [-1.0, 0.5], "gamma_flip": 6600.0, "close:SPX": 6500.0}))
        self.assertIs(result.status, Status.NOT_MET)
        self.assertIn("below the flip", result.summary)

    def test_index_above_the_flip_level_is_met(self):
        result = run(make_store({"gex": [-1.0, 0.5], "gamma_flip": 6400.0, "close:SPX": 6500.0}))
        self.assertIs(result.status, Status.MET)

    def test_strong_dix_is_flagged(self):
        result = run(make_store({"gex": [-1.0, 1.0], "dix": 46.2}))
        self.assertTrue(any("dark-pool" in f for f in result.flags))

    def test_missing_gex_is_unknown(self):
        self.assertIs(run(make_store({"dix": 44.0})).status, Status.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
