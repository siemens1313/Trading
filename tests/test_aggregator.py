import unittest
from datetime import timedelta

from bottom_bots.bots import SENSOR_BOTS, AggregatorBot
from bottom_bots.history import RunRecord
from bottom_bots.models import BotResult, Status
from tests.helpers import AS_OF, cfg

CODES = {"M": Status.MET, "N": Status.NOT_MET, "U": Status.UNKNOWN}


def results(codes: str) -> list[BotResult]:
    return [
        BotResult(key=b.key, item=b.item, title=b.title, rule=b.rule, status=CODES[c], summary="")
        for b, c in zip(SENSOR_BOTS, codes)
    ]


def past(score: int, persistent: int | None = None, target: float = 0.0, days_ago: int = 1) -> RunRecord:
    return RunRecord(
        as_of=AS_OF - timedelta(days=days_ago),
        score=score,
        unknown=0,
        persistent=score if persistent is None else persistent,
        target=target,
        increment=0.0,
    )


def judge(codes: str, previous=(), **sections):
    return AggregatorBot(cfg(**sections)).evaluate(results(codes), list(previous), AS_OF)


class AggregatorTest(unittest.TestCase):
    def test_regimes_follow_the_article(self):
        expected = {
            "NNNNN": "NO SIGNAL",
            "MNNNN": "OVERSOLD ONLY",
            "MMNNN": "OVERSOLD ONLY",
            "MMMNN": "BUILDING",
            "MMMMN": "CAPITULATION-LIKE",
            "MMMMM": "CAPITULATION-LIKE",
        }
        for codes, regime in expected.items():
            self.assertEqual(judge(codes).regime, regime, codes)

    def test_unknown_items_raise_the_ceiling_not_the_score(self):
        verdict = judge("MUUNN")
        self.assertEqual((verdict.score, verdict.unknown, verdict.ceiling), (1, 2, 3))

    def test_a_first_run_never_deploys(self):
        plan = judge("MMMMM").plan
        self.assertEqual((plan.persistent_score, plan.target, plan.runs_needed), (0, 0.0, 1))

    def test_a_score_that_held_deploys_its_tranche(self):
        plan = judge("MMMMM", [past(5)]).plan
        self.assertEqual((plan.persistent_score, plan.target, plan.increment), (5, 0.75, 0.75))

    def test_the_persistent_score_is_the_weakest_recent_run(self):
        plan = judge("MMMMM", [past(3)]).plan
        self.assertEqual((plan.persistent_score, plan.target, plan.runs_needed), (3, 0.25, 1))

    def test_money_back_gate_holds_four_without_flows(self):
        plan = judge("MMMMN", [past(4)]).plan
        self.assertTrue(plan.gated)
        self.assertEqual(plan.target, 0.25)

    def test_gate_can_be_turned_off(self):
        plan = judge("MMMMN", [past(4)], aggregator={"money_back_gate": False}).plan
        self.assertFalse(plan.gated)
        self.assertEqual(plan.target, 0.50)

    def test_targets_ratchet_up_and_pause_when_conditions_weaken(self):
        plan = judge("MNNNN", [past(4, persistent=4, target=0.5)]).plan
        self.assertEqual((plan.target, plan.increment), (0.5, 0.0))
        self.assertTrue(plan.paused)

    def test_orders_split_the_tranche_by_weight(self):
        plan = judge("MMMNN", [past(3)], plan={"reserve_cash": 10_000.0}).plan
        self.assertEqual(
            plan.orders,
            [
                {"symbol": "QQQ", "side": "BUY", "notional": 1250.0},
                {"symbol": "RSP", "side": "BUY", "notional": 1250.0},
            ],
        )

    def test_no_orders_without_a_reserve(self):
        plan = judge("MMMNN", [past(3)]).plan
        self.assertEqual((plan.increment, plan.orders), (0.25, []))

    def test_longer_persistence_needs_more_runs(self):
        plan = judge("MMMMM", [past(5, days_ago=1)], aggregator={"persistence_runs": 3}).plan
        self.assertEqual((plan.persistent_score, plan.runs_needed), (0, 1))
        plan = judge("MMMMM", [past(5, days_ago=2), past(5, days_ago=1)], aggregator={"persistence_runs": 3}).plan
        self.assertEqual(plan.persistent_score, 5)


if __name__ == "__main__":
    unittest.main()
