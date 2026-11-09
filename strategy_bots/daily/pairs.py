"""Strategy 1: pairs trading.

Formation, then trading: every `retest_every` days the bot fits the hedge ratio (OLS of log A on log B)
over the last `formation` days and tests the spread for cointegration (Engle-Granger t-statistic below
-3.34, the 5% critical value). Until the next retest, a pair that passed is traded with that fixed fit:
when the spread is stretched more than entry_z of its formation standard deviation, go long the cheap
leg and short the rich one; exit when it reverts to exit_z, blows out to stop_z, after max_hold days, or
when a retest fails. Testing before the stretch matters: a test run on a window that includes the
divergence tends to reject the very trades it should allow. Needs short selling, which Robinhood
doesn't offer; it's a paper strategy unless your broker allows shorts.
"""

from __future__ import annotations

import numpy as np

from .base import DailyStrategy


def hedge(ya: np.ndarray, yb: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS of ya on yb: (beta, alpha, residuals)."""
    xm, ym = yb.mean(), ya.mean()
    sxx = ((yb - xm) ** 2).sum()
    beta = float(((yb - xm) * (ya - ym)).sum() / sxx) if sxx > 0 else 0.0
    alpha = float(ym - beta * xm)
    return beta, alpha, ya - alpha - beta * yb


def adf_t(resid: np.ndarray) -> float:
    """t-statistic of rho in d(e_t) = c + rho * e_(t-1): strongly negative means the spread mean-reverts."""
    dy, x = np.diff(resid), resid[:-1]
    xm, ym = x.mean(), dy.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0 or len(dy) < 3:
        return 0.0
    rho = ((x - xm) * (dy - ym)).sum() / sxx
    e = dy - (ym - rho * xm) - rho * x
    return float(rho / np.sqrt((e**2).sum() / (len(dy) - 2) / sxx))


class PairsTrading(DailyStrategy):
    name, number, title = "pairs", 1, "Pairs trading (cointegrated spreads)"
    long_only = False
    defaults = {
        "pairs": [["KO", "PEP"], ["V", "MA"], ["XOM", "CVX"]],
        "formation": 250,
        "retest_every": 21,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 4.0,
        "max_hold": 30,
        "adf_t": -3.34,
    }

    def warmup(self):
        return self.p["formation"]

    def weights(self, panel):
        p = self.p
        T, N = panel.close.shape
        W = np.zeros((T, N))
        pairs = [(a, b) for a, b in p["pairs"] if a in panel.symbols and b in panel.symbols]
        if not pairs:
            return W
        gross = 1 / len(pairs)
        logc = np.log(panel.close)
        n = p["formation"]
        self.models: dict[str, tuple[float, float, float] | None] = {}
        for a, b in pairs:
            ia, ib = panel.col(a), panel.col(b)
            ya, yb = logc[:, ia], logc[:, ib]
            state, held, model = 0, 0, None
            for t in range(n - 1, T):
                if (t - n + 1) % p["retest_every"] == 0:
                    beta, alpha, resid = hedge(ya[t - n + 1 : t + 1], yb[t - n + 1 : t + 1])
                    sd = resid.std()
                    model = (beta, alpha, sd) if sd > 0 and adf_t(resid) < p["adf_t"] else None
                    if model is None:
                        state = 0  # no longer cointegrated: close the trade
                if model is None:
                    continue
                beta, alpha, sd = model
                z = (ya[t] - alpha - beta * yb[t]) / sd
                if state == 0:
                    if abs(z) >= p["entry_z"]:
                        state, held = (-1 if z > 0 else 1), 0
                else:
                    held += 1
                    if abs(z) <= p["exit_z"] or abs(z) >= p["stop_z"] or held >= p["max_hold"] or z * state > 0:
                        state = 0
                if state:
                    scale = gross / (1 + abs(beta))
                    W[t, ia] += state * scale
                    W[t, ib] -= state * beta * scale
            self.models[f"{a}/{b}"] = model
        return W

    def describe(self, panel):
        logc = np.log(panel.close[-1])
        lines = []
        for pair, model in self.models.items():
            if model is None:
                lines.append(f"{pair}: not cointegrated at the last test, no trade")
                continue
            a, b = pair.split("/")
            beta, alpha, sd = model
            z = (logc[panel.col(a)] - alpha - beta * logc[panel.col(b)]) / sd
            lines.append(f"{pair}: cointegrated, hedge ratio {beta:.2f}, spread z {z:+.2f} (trades beyond +/-{self.p['entry_z']})")
        return lines
