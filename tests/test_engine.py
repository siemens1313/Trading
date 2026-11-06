from datetime import datetime

import pytest

from conftest import synth_bars
from quantbot.data import DataProvider
from quantbot.engine import NY, EngineState, make_broker, market_open, run_cycle


class FakeProvider(DataProvider):
    def __init__(self, bars, price_mult=1.0):
        self.bars, self.mult = bars, price_mult

    def history(self, symbols):
        return {s: self.bars[s] for s in symbols}

    def latest_prices(self, symbols):
        return {s: float(self.bars[s]["close"].iloc[-1]) * self.mult for s in symbols}


def bullish(cfg):
    return synth_bars(cfg.universe.symbols, n=600, drifts={s: 0.001 for s in cfg.universe.symbols})


def test_rebalances_once_per_period(cfg):
    prov = FakeProvider(bullish(cfg))
    now = datetime(2030, 3, 4, 10, 0, tzinfo=NY)
    r1 = run_cycle(cfg, provider=prov, now=now)
    assert r1.action == "rebalance" and r1.fills and all(f.status == "filled" for f in r1.fills)
    r2 = run_cycle(cfg, provider=prov, now=now.replace(day=5))
    assert r2.action.startswith("holding") and not r2.fills
    r3 = run_cycle(cfg, provider=prov, now=now.replace(month=4, day=1))
    assert r3.action == "rebalance"


def test_dry_run_changes_nothing(cfg):
    prov = FakeProvider(bullish(cfg))
    r = run_cycle(cfg, provider=prov, dry_run=True, now=datetime(2030, 3, 4, 10, tzinfo=NY))
    assert r.orders and not r.fills
    assert not (cfg.path("state") / "engine_state.json").exists()


def test_kill_switch_flattens_and_halts(cfg):
    bars = bullish(cfg)
    now = datetime(2030, 3, 4, 10, tzinfo=NY)
    run_cycle(cfg, provider=FakeProvider(bars), now=now)
    cfg.risk.max_price_deviation = 1.0  # let the crash quote through
    r = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.5), now=now.replace(day=5))
    assert r.action.startswith("KILL SWITCH")
    assert make_broker(cfg).positions() == {}
    assert EngineState.load(cfg.path("state") / "engine_state.json").halted
    assert run_cycle(cfg, provider=FakeProvider(bars), now=now.replace(day=6)).action.startswith("halted")


def test_live_mode_is_locked_by_default(cfg, monkeypatch):
    monkeypatch.delenv("QUANTBOT_ALLOW_LIVE", raising=False)
    cfg.general.mode = "live"
    with pytest.raises(RuntimeError, match="locked"):
        make_broker(cfg)


def test_market_hours():
    assert market_open(datetime(2026, 11, 6, 10, 0, tzinfo=NY))
    assert not market_open(datetime(2026, 11, 6, 16, 0, tzinfo=NY))
    assert not market_open(datetime(2026, 11, 7, 12, 0, tzinfo=NY))  # Saturday
