import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbot.config import AlphaCfg, Config, UniverseCfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def synth_bars(symbols=SYMS, n=700, seed=0, drifts=None, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    mkt = rng.normal(0.0003, vol, n)
    out = {}
    for i, s in enumerate(symbols):
        mu = (drifts or {}).get(s, 0.0002 * (i - 2))
        r = mu + 0.8 * mkt + rng.normal(0, vol * 0.6, n)
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.002, n))
        out[s] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                               "low": np.minimum(open_, close) * 0.998, "close": close, "volume": 1e7}, index=idx)
    return out


@pytest.fixture
def cfg(tmp_path):
    c = Config(universe=UniverseCfg(symbols=SYMS, benchmark="AAA"), root=tmp_path,
               alphas=AlphaCfg(ic_window=126, ic_min_obs=40,
                               params={"tsmom": {"lookback": 120, "skip": 10}, "xsmom": {"lookback": 60, "skip": 5},
                                       "strev": {"lookback": 5, "beta_window": 40}, "lowvol": {"window": 40}}))
    c.portfolio.beta_window = 60
    c.risk_model.corr_window = 60
    return c
