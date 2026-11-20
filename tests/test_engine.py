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


NOW = datetime(2030, 3, 4, 10, 0, tzinfo=NY)  # after all synthetic data


def test_rebalances_once_per_day_and_journals(cfg):
    prov = FakeProvider(synth_bars(n=700, seed=1))
    r1 = run_cycle(cfg, provider=prov, now=NOW)
    assert r1.action == "rebalance" and r1.fills and all(f.status == "filled" for f in r1.fills)
    assert r1.decision is not None and r1.decision.weights.sum() > 0
    r2 = run_cycle(cfg, provider=prov, now=NOW.replace(hour=11))
    assert r2.action.startswith("holding") and not r2.fills
    r3 = run_cycle(cfg, provider=prov, now=NOW.replace(day=5))
    assert r3.action == "rebalance"
    journal = (cfg.path("state") / "journal.jsonl").read_text().splitlines()
    assert len(journal) == 2


def test_paper_fills_cross_the_spread(cfg):
    bars = synth_bars(n=700, seed=1)
    r = run_cycle(cfg, provider=FakeProvider(bars), now=NOW)
    f = next(f for f in r.fills if f.side == "buy")
    last = bars[f.symbol]["close"].iloc[-1]
    assert f.price > last * (1 + cfg.costs.half_spread_bps / 1e4)  # paid the ask plus slippage


def test_dry_run_changes_nothing(cfg):
    r = run_cycle(cfg, provider=FakeProvider(synth_bars(n=700, seed=1)), dry_run=True, now=NOW)
    assert r.orders and not r.fills
    assert not (cfg.path("state") / "engine_state.json").exists()


def test_kill_switch_flattens_and_halts(cfg):
    bars = synth_bars(n=700, seed=1)
    run_cycle(cfg, provider=FakeProvider(bars), now=NOW)
    cfg.risk.max_price_deviation = 1.0
    cfg.risk.daily_loss_limit = 1.0
    r = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.3), now=NOW.replace(day=5))
    assert r.action.startswith("KILL SWITCH")
    assert make_broker(cfg).positions() == {}
    assert EngineState.load(cfg.path("state") / "engine_state.json").halted
    assert run_cycle(cfg, provider=FakeProvider(bars), now=NOW.replace(day=6)).action.startswith("halted")


def test_daily_loss_limit_blocks_trading(cfg):
    bars = synth_bars(n=700, seed=1)
    run_cycle(cfg, provider=FakeProvider(bars), now=NOW)
    cfg.risk.max_price_deviation = 1.0
    r = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.9), now=NOW.replace(hour=12), force=True)
    assert r.action.startswith("daily loss limit") and not r.fills


def test_live_mode_is_locked_and_long_only(cfg, monkeypatch):
    monkeypatch.delenv("QUANTBOT_ALLOW_LIVE", raising=False)
    cfg.general.mode = "live"
    with pytest.raises(RuntimeError, match="locked"):
        make_broker(cfg)
    cfg.portfolio.long_only = False
    with pytest.raises(ValueError, match="long_only"):
        cfg.validate()


def test_market_hours():
    assert market_open(datetime(2026, 11, 6, 10, 0, tzinfo=NY))
    assert not market_open(datetime(2026, 11, 6, 16, 0, tzinfo=NY))
    assert not market_open(datetime(2026, 11, 7, 12, 0, tzinfo=NY))


def test_daily_loss_limit_measures_from_previous_run(cfg):
    """Regression: with one run per day the limit used to compare today's equity with itself (always 0%).
    A loss since yesterday's run must now block trading on the first run of the new day."""
    bars = synth_bars(n=700, seed=1)
    run_cycle(cfg, provider=FakeProvider(bars), now=NOW)                    # day 1: invest
    cfg.risk.max_price_deviation = 1.0
    r = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.95), now=NOW.replace(day=5))  # next day, -5%
    assert r.day_pnl < -cfg.risk.daily_loss_limit
    assert r.action.startswith("daily loss limit") and not r.fills
    r2 = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.95), now=NOW.replace(day=6))  # flat vs. yesterday
    assert abs(r2.day_pnl) < 1e-9 and r2.action == "rebalance"


def test_daily_loss_reference_falls_back_to_equity_log(cfg):
    """State files written before last_equity existed fall back to the last row of the equity log."""
    bars = synth_bars(n=700, seed=1)
    r1 = run_cycle(cfg, provider=FakeProvider(bars), now=NOW)
    p = cfg.path("state") / "engine_state.json"
    st = EngineState.load(p)
    st.last_equity = None
    st.save()
    cfg.risk.max_price_deviation = 1.0
    r2 = run_cycle(cfg, provider=FakeProvider(bars, price_mult=0.98), now=NOW.replace(day=5), dry_run=True)
    assert r2.day_pnl < -0.01                                   # the old behaviour would report 0% here
    assert r2.day_pnl == pytest.approx(r2.equity / round(r1.equity, 2) - 1, abs=1e-9)  # log keeps cents
