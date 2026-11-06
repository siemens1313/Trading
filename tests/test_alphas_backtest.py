import numpy as np
import pandas as pd

from conftest import ROOT, synth_bars
from quantbot import alphas as A
from quantbot.backtest import build_panel, run_backtest
from quantbot.config import load_config
from quantbot.data import CSVProvider, align


def _shock_after(bars, cut, seed=99):
    rng = np.random.default_rng(seed)
    out = {}
    for s, df in bars.items():
        df = df.copy()
        m = df.index > cut
        df.loc[m, ["open", "high", "low", "close"]] *= rng.uniform(0.5, 1.5, size=(int(m.sum()), 1))
        out[s] = df
    return out


def test_forecast_has_no_lookahead(cfg):
    """mu, alpha weights and ICs at date D must not change when anything after D changes.
    This is the test that catches IC-lag bugs: forward returns are used to measure ICs."""
    bars = synth_bars(n=700, seed=3)
    cut = bars["AAA"].index[550]
    p1, p2 = build_panel(bars, cfg), build_panel(_shock_after(bars, cut), cfg)
    pd.testing.assert_frame_equal(p1.fc.mu.loc[:cut], p2.fc.mu.loc[:cut])
    pd.testing.assert_frame_equal(p1.fc.alpha_weights.loc[:cut], p2.fc.alpha_weights.loc[:cut])
    pd.testing.assert_series_equal(p1.fc.combined_ic.loc[:cut], p2.fc.combined_ic.loc[:cut])
    assert p1.fc.combined_ic.loc[:cut].gt(0).any()  # the test is vacuous unless sizing is actually on


def test_backtest_has_no_lookahead(cfg):
    bars = synth_bars(n=700, seed=4)
    cut = bars["AAA"].index[600]
    a = run_backtest(bars, cfg).equity
    b = run_backtest(_shock_after(bars, cut), cfg).equity
    pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])


def test_combiner_rewards_real_edge_and_ignores_anti_edge(cfg, monkeypatch):
    """Plant one alpha that genuinely predicts (peeks at future returns + noise) and one that predicts
    backwards. The combiner must learn, from realized ICs only, to weight the first and zero the second."""
    bars = synth_bars(n=700, seed=5)
    O = align(bars, "open")
    fwd = A.forward_returns(O, cfg.alphas.horizon)
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    good = lambda C, bench, **k: (fwd / fwd.std() + 3 * noise).reindex(C.index)  # noqa: E731
    bad = lambda C, bench, **k: (-fwd / fwd.std() + 3 * noise).reindex(C.index)  # noqa: E731
    monkeypatch.setitem(A.ALPHAS, "good", good)
    monkeypatch.setitem(A.ALPHAS, "bad", bad)
    cfg.alphas.enabled = ["good", "bad"]
    fc = build_panel(bars, cfg).fc
    late = fc.alpha_weights.iloc[-150:]
    assert (late["bad"] == 0).all()
    assert (late["good"] > 0.99).all()
    assert fc.combined_ic.iloc[-150:].gt(0).all()
    # and nothing is weighted before enough realized history exists
    first = fc.alpha_weights.index[(fc.alpha_weights.sum(axis=1) > 0).argmax()]
    assert fc.alpha_weights.index.get_loc(first) >= cfg.alphas.ic_min_obs


def test_backtest_accounting_and_limits(cfg):
    res = run_backtest(synth_bars(n=700, seed=6), cfg)
    assert (res.equity > 0).all()
    assert (res.weights.abs().sum(axis=1) <= cfg.portfolio.max_gross + 0.03).all()
    assert (res.weights >= -1e-9).all().all()                        # long-only by default
    assert (res.weights <= cfg.portfolio.max_weight + 0.03).all().all()
    assert res.metrics["costs_total"] > 0


def test_long_short_paper_book(cfg):
    cfg.portfolio.long_only = False
    res = run_backtest(synth_bars(n=700, seed=7), cfg)
    assert (res.weights < -1e-4).any().any()                         # actually shorts something
    assert (res.weights.abs().sum(axis=1) <= cfg.portfolio.max_gross + 0.03).all()


def test_real_data_pipeline_runs():
    cfg = load_config(ROOT / "config.toml")
    res = run_backtest(CSVProvider(ROOT / "data").history(cfg.universe.symbols), cfg)
    assert res.metrics["max_drawdown"] > -0.5 and len(res.trades) > 0
    assert 0 <= res.metrics["avg_gross"] <= 1.0001
