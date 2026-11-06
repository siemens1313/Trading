import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbot.config import Config, UniverseCfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def synth_bars(symbols, n=600, seed=0, drifts=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for i, s in enumerate(symbols):
        mu = (drifts or {}).get(s, 0.0004 * (i - 1))
        r = rng.normal(mu, 0.01, n)
        close = 100 * np.exp(np.cumsum(r))
        open_ = close * np.exp(rng.normal(0, 0.002, n))
        out[s] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                               "low": np.minimum(open_, close) * 0.998, "close": close,
                               "volume": 1e6}, index=idx)
    return out


@pytest.fixture
def cfg(tmp_path):
    c = Config(universe=UniverseCfg(symbols=["AAA", "BBB", "CCC", "DDD"], benchmark="AAA"), root=tmp_path)
    c.general.data_provider = "csv"
    return c
