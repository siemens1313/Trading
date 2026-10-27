import unittest

from bottom_bots.bots.short_interest import ShortInterestBot
from bottom_bots.models import Status
from tests.helpers import AS_OF, cfg, make_store


def run(store):
    return ShortInterestBot(cfg(), store, AS_OF).evaluate()


def v_shape(sessions_ago: int, n: int = 60) -> list[float]:
    """Closes falling into a low `sessions_ago` sessions before the last one, then recovering."""
    bottom = n - 1 - sessions_ago
    return [100.0 + abs(bottom - i) for i in range(n)]


class ShortInterestBotTest(unittest.TestCase):
    def test_heavy_shorts_and_no_new_low_is_met(self):
        result = run(make_store({"close:SOXX": v_shape(8), "si_pct_float": 12.0}))
        self.assertIs(result.status, Status.MET)
        self.assertIn("8 sessions old", result.summary)

    def test_new_low_today_is_not_met(self):
        self.assertIs(run(make_store({"close:SOXX": v_shape(0), "si_pct_float": 12.0})).status, Status.NOT_MET)

    def test_light_short_interest_is_not_met(self):
        store = make_store({"close:SOXX": v_shape(8), "si_pct_float": 4.0, "days_to_cover": 2.0})
        self.assertIs(run(store).status, Status.NOT_MET)

    def test_days_to_cover_alone_counts_as_heavy(self):
        self.assertIs(run(make_store({"close:SOXX": v_shape(8), "days_to_cover": 6.5})).status, Status.MET)

    def test_missing_short_interest_is_unknown(self):
        self.assertIs(run(make_store({"close:SOXX": v_shape(8)})).status, Status.UNKNOWN)

    def test_crowded_and_ignition_are_flagged(self):
        flags = " ".join(run(make_store({"close:SOXX": v_shape(8), "si_pct_float": 22.0})).flags)
        self.assertIn("crowded", flags)
        self.assertIn("Ignition", flags)


if __name__ == "__main__":
    unittest.main()
