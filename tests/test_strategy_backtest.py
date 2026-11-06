import numpy as np
import pandas as pd

from conftest import ROOT, synth_bars
from quantbot.backtest import run_backtest
from quantbot.config import load_config
from quantbot.data import CSVProvider, align
from quantbot.strategy import compute_targets, required_history


def test_not_enough_history_returns_none(cfg):
    closes = align(synth_bars(cfg.universe.symbols, n=100), "close")
    assert compute_targets(closes, cfg.strategy) is None


def test_weights_respect_caps_and_top_n(cfg):
    closes = align(synth_bars(cfg.universe.symbols, n=600, drifts={"AAA": .001, "BBB": .001, "CCC": .001, "DDD": .001}), "close")
    w = compute_targets(closes, cfg.strategy)
    assert (w >= 0).all()
    assert w.sum() <= 1 + 1e-9
    assert (w <= cfg.strategy.max_weight + 1e-12).all()
    assert (w > 0).sum() <= cfg.strategy.top_n


def test_all_cash_in_a_downtrend(cfg):
    closes = align(synth_bars(cfg.universe.symbols, n=600, drifts={s: -0.002 for s in cfg.universe.symbols}), "close")
    assert compute_targets(closes, cfg.strategy).sum() == 0


def test_no_lookahead(cfg):
    """Changing prices after date D must not change anything up to D."""
    bars = synth_bars(cfg.universe.symbols, n=700, seed=3)
    cut = bars["AAA"].index[500]
    base = run_backtest(bars, cfg).equity
    rng = np.random.default_rng(99)
    shocked = {}
    for s, df in bars.items():
        df = df.copy()
        m = df.index > cut
        df.loc[m, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 1.5, size=(m.sum(), 1))
        shocked[s] = df
    alt = run_backtest(shocked, cfg).equity
    pd.testing.assert_series_equal(base.loc[:cut], alt.loc[:cut])


def test_backtest_accounting_is_sane(cfg):
    res = run_backtest(synth_bars(cfg.universe.symbols, n=700, seed=5), cfg)
    assert (res.equity > 0).all()
    assert (res.weights.sum(axis=1) <= 1 + 1e-6).all()   # never levered
    assert (res.weights >= -1e-12).all().all()          # never short


def test_real_data_backtest_runs():
    cfg = load_config(ROOT / "config.toml")
    bars = CSVProvider(ROOT / "data").history(cfg.universe.symbols)
    res = run_backtest(bars, cfg)
    assert res.metrics["max_drawdown"] > -0.5
    assert len(res.trades) > 0
