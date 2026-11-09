import unittest

from bottom_bots.bots.flows_sentiment import FlowsSentimentBot
from bottom_bots.models import Status
from tests.helpers import AS_OF, cfg, make_store

TURNED = [-100, -200, -50, -80, 30, 40, 60]


def run(store, **sections):
    return FlowsSentimentBot(cfg(**sections), store, AS_OF).evaluate()


class FlowsSentimentBotTest(unittest.TestCase):
    def test_flows_turn_and_aaii_rollover_is_met(self):
        store = make_store(weekly={"etf_flows_weekly": TURNED, "aaii_bearish": [40, 48, 55, 52]})
        self.assertIs(run(store).status, Status.MET)

    def test_article_reading_is_not_met_but_sentiment_is_extreme(self):
        store = make_store({"fear_greed": 21, "naaim": 87.7, "breadth_200dma": 57}, weekly={"aaii_bearish": 45.8})
        result = run(store)
        self.assertIs(result.status, Status.NOT_MET)
        self.assertEqual(len(result.extremes), 1)
        self.assertIn("Fear & Greed 21", result.extremes[0])
        flags = " ".join(result.flags)
        self.assertIn("NAAIM 87.7", flags)
        self.assertIn("no broad washout", flags)

    def test_aaii_still_rising_is_not_met(self):
        store = make_store(weekly={"etf_flows_weekly": TURNED, "aaii_bearish": [45, 52, 56]})
        result = run(store)
        self.assertIs(result.status, Status.NOT_MET)
        self.assertIn("isn't falling yet", result.summary)

    def test_aaii_rebounding_below_its_peak_is_not_falling(self):
        store = make_store(weekly={"etf_flows_weekly": TURNED, "aaii_bearish": [45, 58, 50, 53]})
        self.assertIs(run(store).status, Status.NOT_MET)

    def test_two_weeks_of_inflows_is_not_enough(self):
        store = make_store(weekly={"etf_flows_weekly": [-100, -200, -50, -80, 40, 60], "aaii_bearish": [55, 52]})
        self.assertIs(run(store).status, Status.NOT_MET)

    def test_inflows_without_prior_outflows_are_not_a_turn(self):
        store = make_store(weekly={"etf_flows_weekly": [10, 20, 30, 40, 50, 60, 70], "aaii_bearish": [55, 52]})
        self.assertIs(run(store).status, Status.NOT_MET)

    def test_require_any_accepts_flows_alone(self):
        store = make_store(weekly={"etf_flows_weekly": TURNED, "aaii_bearish": [40, 42]})
        self.assertIs(run(store, flows_sentiment={"require": "any"}).status, Status.MET)

    def test_missing_flows_with_an_aaii_rollover_is_unknown(self):
        self.assertIs(run(make_store(weekly={"aaii_bearish": [40, 55, 51]})).status, Status.UNKNOWN)

    def test_washout_rare_fear_and_aaii_extreme(self):
        result = run(make_store({"fear_greed": 8, "breadth_200dma": 15}, weekly={"aaii_bearish": 60}))
        text = " ".join(result.extremes)
        self.assertIn("rare", text)
        self.assertIn("washout", text)
        self.assertIn("AAII bearish 60.0%", text)


if __name__ == "__main__":
    unittest.main()
