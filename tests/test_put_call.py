import unittest

from bottom_bots.bots.put_call import PutCallBot
from bottom_bots.models import Status
from tests.helpers import AS_OF, cfg, make_store


def run(store):
    return PutCallBot(cfg(), store, AS_OF).evaluate()


class PutCallBotTest(unittest.TestCase):
    def test_article_reading_is_not_met(self):
        result = run(make_store({"pc_equity": 0.67, "pc_total": 0.93, "pc_index": 1.10}))
        self.assertIs(result.status, Status.NOT_MET)
        self.assertEqual([r.value for r in result.readings], [0.67, 0.93, 1.10])

    def test_threshold_is_inclusive(self):
        self.assertIs(run(make_store({"pc_equity": 0.80})).status, Status.MET)

    def test_euphoria_is_flagged(self):
        result = run(make_store({"pc_equity": 0.45}))
        self.assertIs(result.status, Status.NOT_MET)
        self.assertTrue(any("overheated" in f for f in result.flags))

    def test_percentile_against_the_previous_year(self):
        result = run(make_store({"pc_equity": [0.5] * 85 + [0.9] * 15 + [0.67]}))
        self.assertIn("above 85% of the previous 100 sessions", " ".join(result.flags))

    def test_no_percentile_with_short_history(self):
        result = run(make_store({"pc_equity": [0.5] * 10 + [0.67]}))
        self.assertFalse(any("previous" in f for f in result.flags))

    def test_missing_ratio_is_unknown(self):
        self.assertIs(run(make_store({"pc_total": 0.9})).status, Status.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
