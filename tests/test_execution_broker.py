import pandas as pd
import pytest

from quantbot.broker import PaperBroker, SimulatedBroker
from quantbot.config import CostCfg, ExecutionCfg, RiskCfg
from quantbot.costs import fill_cost_bps
from quantbot.execution import Order, plan_orders
from quantbot.risk import RiskManager
from quantbot.stats import bootstrap_sharpe_ci, deflated_sharpe, probabilistic_sharpe, sharpe


def test_plan_orders_sells_first_and_closes_dropped_names():
    orders = plan_orders(pd.Series({"A": 0.5, "B": 0.0}), {"B": 10, "C": 5}, {"A": 100, "B": 50, "C": 20}, 10_000,
                         ExecutionCfg())
    assert [o.side for o in orders][:2] == ["sell", "sell"]
    assert {(o.symbol, o.side) for o in orders} == {("A", "buy"), ("B", "sell"), ("C", "sell")}
    assert next(o for o in orders if o.symbol == "A").qty == pytest.approx(50)


def test_plan_orders_no_trade_band_and_whole_shares():
    assert plan_orders(pd.Series({"A": 0.503}), {"A": 50}, {"A": 100}, 10_000, ExecutionCfg(min_trade_weight=0.005)) == []
    (o,) = plan_orders(pd.Series({"A": 0.5}), {}, {"A": 333}, 10_000, ExecutionCfg(fractional=False))
    assert o.qty == 15


def test_plan_orders_handles_shorts():
    (o,) = plan_orders(pd.Series({"A": -0.2}), {}, {"A": 100}, 10_000, ExecutionCfg())
    assert o.side == "sell" and o.qty == pytest.approx(20)
    (o,) = plan_orders(pd.Series({"A": 0.1}), {"A": -20}, {"A": 100}, 10_000, ExecutionCfg())
    assert o.side == "buy" and o.qty == pytest.approx(30)


def test_sim_broker_long_only_vs_short():
    b = SimulatedBroker(1000)
    f = b.submit(Order("A", "buy", 100, 100), 100, cost_bps=10)
    assert f.price == pytest.approx(100.1) and b.cash >= 0      # scaled to available cash
    held = b.positions()["A"]
    assert b.submit(Order("A", "sell", held * 3, 100), 100).qty == pytest.approx(held)
    assert b.submit(Order("Z", "sell", 1, 10), 10).status == "rejected"
    s = SimulatedBroker(1000, allow_short=True)
    s.submit(Order("A", "sell", 5, 100), 100, cost_bps=0)
    assert s.positions()["A"] == pytest.approx(-5) and s.equity({"A": 90}) == pytest.approx(1050)


def test_paper_broker_persists(tmp_path):
    b = PaperBroker(tmp_path, 5000, 0, 0, True)
    b.submit(Order("A", "buy", 10, 100), 100, cost_bps=0)
    b2 = PaperBroker(tmp_path, 999999, 0, 0, True)
    assert b2.cash == pytest.approx(4000) and b2.positions() == {"A": 10}


def test_cost_model():
    c = CostCfg(half_spread_bps=1, slippage_bps=2, impact_coef=0.1)
    assert fill_cost_bps(0, 1e9, 0.01, c) == pytest.approx(3)
    big = fill_cost_bps(1e8, 1e9, 0.01, c)     # 10% of ADV: sqrt(0.1)*0.01*0.1 = 3.2 bps impact
    assert big == pytest.approx(3 + 0.1 * 0.01 * (0.1 ** 0.5) * 1e4)
    assert fill_cost_bps(1e4, 1e9, 0.01, c, crossing_spread=False) < fill_cost_bps(1e4, 1e9, 0.01, c)


def test_risk_checks():
    rm = RiskManager(RiskCfg(max_order_notional=1000, max_price_deviation=0.1, kill_switch_drawdown=0.3))
    ok, bad = rm.check_orders([Order("A", "buy", 1, 150), Order("B", "buy", 100, 50), Order("C", "buy", 1, 10)],
                              {"A": 150, "B": 50, "C": 10}, {"A": 100, "B": 50, "C": 10})
    assert [o.symbol for o in ok] == ["C"] and {o.symbol for o, _ in bad} == {"A", "B"}
    assert not rm.kill_switch(71, 100) and rm.kill_switch(69, 100)


def test_stats_behave():
    import numpy as np
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 1500))
    noise = pd.Series(rng.normal(0.0, 0.01, 1500))
    noise -= noise.mean()                                   # exactly zero Sharpe -> exactly a coin flip
    assert probabilistic_sharpe(good) > 0.99
    assert probabilistic_sharpe(noise) == pytest.approx(0.5)
    lo, hi = bootstrap_sharpe_ci(good)
    assert lo < sharpe(good) < hi
    assert deflated_sharpe(good, 100, 0.5) < probabilistic_sharpe(good)
