"""Strategy 10: machine learning.

Ten causal features per stock and day (returns over 1-63 days, volatility, RSI, distance from the 50- and
200-day averages, volume surprise, Bollinger position) feed a random forest, a neural network and a linear
SVM, whose probabilities are averaged. The label is whether the stock rises from the next open over
`horizon` days. Models are retrained every `retrain_every` days, walk-forward, on samples whose label was
already known: a sample from day s is used only from day s + 1 + horizon on.
"""

from __future__ import annotations

import numpy as np

from ..indicators import bollinger, rolling_std, rsi, sma
from ..learn import MLP, LinearSVM, RandomForest, Standardizer
from .base import DailyStrategy, first_fit, log_returns, slots

FEATURES = ("ret_1", "ret_5", "ret_21", "ret_63", "vol_21", "rsi_14", "vs_sma50", "vs_sma200", "volume_z", "bollinger")


def features(panel) -> np.ndarray:
    """(days, symbols, features); NaN until each feature has enough history."""
    c = panel.close

    def ret(n: int) -> np.ndarray:
        out = np.full(c.shape, np.nan)
        out[n:] = np.log(c[n:] / c[:-n])
        return out

    logv = np.log(np.maximum(panel.volume, 1.0))
    _, upper, lower, _ = bollinger(c, 20, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = (logv - sma(logv, 63)) / rolling_std(logv, 63)
        band = (c - lower) / (upper - lower) - 0.5
    cols = [
        ret(1), ret(5), ret(21), ret(63), rolling_std(log_returns(c), 21), rsi(c, 14) / 100 - 0.5,
        c / sma(c, 50) - 1, c / sma(c, 200) - 1, volume_z, band,
    ]
    return np.stack(cols, axis=-1)


class Ensemble:
    def __init__(self, models: list[str], seed: int):
        self.names, self.seed = models, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ensemble:
        self.scaler = Standardizer().fit(X)
        Z = self.scaler.transform(X)
        self.models = []
        for name in self.names:
            if name == "forest":
                self.models.append(RandomForest(seed=self.seed).fit(Z, y))
            elif name == "mlp":
                self.models.append(MLP([Z.shape[1], 16, 1], seed=self.seed).fit_binary(Z, y, seed=self.seed))
            elif name == "svm":
                self.models.append(LinearSVM(seed=self.seed).fit(Z, y))
            else:
                raise ValueError(f"unknown model {name!r}: use forest, mlp or svm")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        return np.mean([m.predict_proba(Z) for m in self.models], axis=0)


class MachineLearning(DailyStrategy):
    name, number, title = "ml", 10, "Machine learning (random forest + neural net + SVM)"
    defaults = {
        "models": ["forest", "mlp", "svm"],
        "horizon": 5,
        "retrain_every": 126,
        "train_days": 756,
        "min_train_days": 252,
        "threshold": 0.53,
        "slots": 5,
        "seed": 7,
    }

    def weights(self, panel):
        p = self.p
        X = features(panel)
        T, N, F = X.shape
        h = p["horizon"]
        o = panel.open
        y = np.full((T, N), np.nan)
        if T > 1 + h:
            y[: T - 1 - h] = (o[1 + h :] > o[1 : T - h]).astype(float)
        W = np.zeros((T, N))
        model, held = None, np.zeros(N)
        self.probabilities = np.full((T, N), np.nan)
        for t in range(T):
            if t % p["retrain_every"] == 0:
                lo, hi = self._window(t)
                if hi - lo >= p["min_train_days"]:
                    Xs, ys = X[lo:hi].reshape(-1, F), y[lo:hi].reshape(-1)
                    ok = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
                    if ok.sum() >= 200 and 0 < ys[ok].mean() < 1:
                        model = Ensemble(p["models"], p["seed"] + t).fit(Xs[ok], ys[ok])
            if model is not None and t % h == 0:
                ok = np.isfinite(X[t]).all(axis=1)
                prob = np.full(N, np.nan)
                if ok.any():
                    prob[ok] = model.predict_proba(X[t][ok])
                self.probabilities[t] = prob
                held = slots(np.nan_to_num(prob) > p["threshold"], p["slots"], np.nan_to_num(prob), held=held > 0)
            W[t] = held
        return W

    def describe(self, panel):
        rows = np.flatnonzero(np.isfinite(self.probabilities).any(axis=1))
        if not len(rows):
            return ["not trained yet: not enough history"]
        t, prob = rows[-1], self.probabilities[rows[-1]]
        ranked = sorted((float(v), s) for s, v in zip(panel.symbols, prob) if np.isfinite(v))[::-1][:8]
        top = ", ".join(f"{s} {v:.2f}" for v, s in ranked)
        return [f"last rebalanced at the {panel.dates[t]} close; P(up over {self.p['horizon']} days), buying above {self.p['threshold']}: {top}"]

    def _window(self, t: int) -> tuple[int, int]:
        """Training rows at day t: samples s <= t-1-horizon, whose labels are known by the close of t."""
        h = self.p["horizon"]
        return max(0, t - 1 - h - self.p["train_days"]), t - h

    def warmup(self) -> int:
        t = first_fit(self.p["retrain_every"], self.p["min_train_days"], self._window)
        return t + (-t) % self.p["horizon"]  # it trades on multiples of the horizon
