import pandas as pd
import pytest

from quantbot.broker import PaperBroker, SimulatedBroker
from quantbot.config import ExecutionCfg, RiskCfg
from quantbot.execution import Order, plan_orders
from quantbot.risk import RiskManager


def test_plan_orders_sells_first_and_closes_dropped_names():
    ex = ExecutionCfg()
    orders = plan_orders(pd.Series({"A": 0.5, "B": 0.0}), {"B": 10, "C": 5}, {"A": 100, "B": 50, "C": 20}, 10_000, ex)
    sides = [o.side for o in orders]
    assert sides == sorted(sides, key=lambda s: s != "sell")
    assert {(o.symbol, o.side) for o in orders} == {("A", "buy"), ("B", "sell"), ("C", "sell")}
    assert next(o for o in orders if o.symbol == "A").qty == pytest.approx(50)


def test_plan_orders_skips_tiny_rebalances():
    ex = ExecutionCfg(min_trade_weight=0.01)
    orders = plan_orders(pd.Series({"A": 0.505}), {"A": 50}, {"A": 100}, 10_000, ex)  # 0.5% drift
    assert orders == []


def test_whole_shares_mode():
    ex = ExecutionCfg(fractional=False)
    (o,) = plan_orders(pd.Series({"A": 0.5}), {}, {"A": 333}, 10_000, ex)
    assert o.qty == 15


def test_sim_broker_no_short_no_negative_cash():
    b = SimulatedBroker(1000, slippage_bps=10)
    f = b.submit(Order("A", "buy", 100, 100), 100)  # wants $10k, has $1k
    assert f.status == "filled" and b.cash >= 0
    assert f.price == pytest.approx(100.1)
    held = b.positions()["A"]
    f = b.submit(Order("A", "sell", held * 3, 100), 100)
    assert f.qty == pytest.approx(held) and "A" not in b.positions()
    assert b.submit(Order("Z", "sell", 1, 10), 10).status == "rejected"


def test_paper_broker_persists(tmp_path):
    b = PaperBroker(tmp_path, 5000, 0, 0, True)
    b.submit(Order("A", "buy", 10, 100), 100)
    b2 = PaperBroker(tmp_path, 999999, 0, 0, True)  # starting cash ignored once an account exists
    assert b2.cash == pytest.approx(4000) and b2.positions() == {"A": 10}
    assert (tmp_path / "paper_trades.csv").read_text().count("\n") == 2


def test_risk_rejects_bad_quotes_and_huge_orders():
    rm = RiskManager(RiskCfg(max_order_notional=1000, max_price_deviation=0.1))
    ok, bad = rm.check_orders([Order("A", "buy", 1, 150), Order("B", "buy", 100, 50), Order("C", "buy", 1, 10)],
                              {"A": 150, "B": 50, "C": 10}, {"A": 100, "B": 50, "C": 10})
    assert [o.symbol for o in ok] == ["C"]
    assert {o.symbol for o, _ in bad} == {"A", "B"}
    assert not rm.kill_switch(71, 100)
    assert rm.kill_switch(69, 100)
