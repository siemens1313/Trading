"""Strategy 13: deep reinforcement learning (DQN).

An agent with a two-hidden-layer Q-network chooses flat or long for one stock at a time, once every
`step_days` days. Its state is the last five daily returns scaled by volatility, RSI, distance from the
50-day average, volatility and its own position; its reward is the open-to-open log return over the next
step, minus costs when it changes position. It learns with experience replay and a target network on
random episodes from the trailing `train_days`, is retrained every `retrain_every` days, and trades the
stocks its greedy policy says to hold. Deciding daily, the policy flipped most positions every day: on
noisy daily returns the two actions' values are nearly equal. Reinforcement learning on stock returns is
notoriously noisy; treat its results with suspicion.
"""

from __future__ import annotations

import math

import numpy as np

from ..indicators import rolling_std, rsi, sma
from ..learn import MLP
from .base import DailyStrategy, first_fit, log_returns, slots


def state_features(panel) -> np.ndarray:
    """(days, symbols, 8) roughly unit-scale features; NaN during warm-up."""
    c = panel.close
    r = log_returns(c)
    vol = rolling_std(r, 21)
    with np.errstate(divide="ignore", invalid="ignore"):
        lags = [np.clip(np.vstack([np.full((k, c.shape[1]), np.nan), r[: len(r) - k]]) / vol, -4, 4) / 2 for k in range(5)]
    return np.stack(
        lags + [(rsi(c, 14) - 50) / 50, (c / sma(c, 50) - 1) * 5, (vol * math.sqrt(252) - 0.3) * 2], axis=-1
    )


class DeepRL(DailyStrategy):
    name, number, title = "rl", 13, "Deep reinforcement learning (DQN)"
    defaults = {
        "retrain_every": 252,
        "train_days": 756,
        "min_train_days": 504,
        "episodes": 40,
        "episode_len": 126,
        "hidden": [32, 16],
        "gamma": 0.9,
        "lr": 1e-3,
        "batch": 64,
        "buffer": 20000,
        "target_sync": 250,
        "epsilon_end": 0.05,
        "cost_bps": 5.0,
        "step_days": 5,
        "slots": 5,
        "seed": 11,
    }

    def weights(self, panel):
        p = self.p
        S = state_features(panel)
        T, N, _ = S.shape
        o = panel.open
        k = p["step_days"]
        rets = np.full((T, N), np.nan)
        if T > k + 1:
            rets[: T - 1 - k] = np.log(o[1 + k :] / o[1 : T - k])  # decided at close s: open s+1 to open s+1+k
        W = np.zeros((T, N))
        net, position = None, np.zeros(N)
        for t in range(T):
            if t % p["retrain_every"] == 0:
                lo, hi = self._window(t)
                if hi - lo >= p["min_train_days"]:
                    net = self._train(S[lo:hi], rets[lo:hi], p["seed"] + t)
            if net is None:
                continue
            if t % k:
                W[t] = W[t - 1]
                continue
            ok = np.isfinite(S[t]).all(axis=1)
            q = net.output(np.hstack([np.nan_to_num(S[t]), position[:, None]]))
            long = (q[:, 1] > q[:, 0]) & ok
            position = long.astype(float)
            W[t] = slots(long, p["slots"], q[:, 1] - q[:, 0], held=W[t - 1] > 0 if t else None)
        return W

    def _window(self, t: int) -> tuple[int, int]:
        """Training rows at day t: s <= t-1-step_days, whose rewards are known by the close of t."""
        return max(0, t - self.p["train_days"]), t - self.p["step_days"]

    def warmup(self) -> int:
        t = first_fit(self.p["retrain_every"], self.p["min_train_days"], self._window)
        return t + (-t) % self.p["step_days"]  # it trades on multiples of the step

    def _train(self, states: np.ndarray, rets: np.ndarray, seed: int) -> MLP:
        p = self.p
        rng = np.random.default_rng(seed)
        L, N, F = states.shape
        net = MLP([F + 1, *p["hidden"], 2], seed=seed, lr=p["lr"])
        target = net.clone()
        size = p["buffer"]
        X, A, R, X2, D = np.zeros((size, F + 1)), np.zeros(size, int), np.zeros(size), np.zeros((size, F + 1)), np.zeros(size)
        filled = ptr = updates = step = 0
        valid = np.isfinite(states).all(axis=2) & np.isfinite(rets)
        step_days = p["step_days"]
        span = p["episode_len"]  # days per episode
        n = max(1, span // step_days)  # decisions per episode
        total = p["episodes"] * n
        cost = p["cost_bps"] / 1e4
        if L <= span + step_days:
            return net
        for _ in range(p["episodes"]):
            for _attempt in range(20):
                j, start = int(rng.integers(N)), int(rng.integers(0, L - n * step_days - 1))
                if valid[start : start + (n + 1) * step_days : step_days, j].all():
                    break
            else:
                continue
            position = 0
            for k in range(n):
                s = start + k * step_days
                x = np.append(states[s, j], position)
                epsilon = max(p["epsilon_end"], 1 - step / (0.6 * total))
                step += 1
                action = int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(net.output(x[None])[0]))
                reward = 100 * (action * rets[s, j] - cost * abs(action - position))
                X[ptr], A[ptr], R[ptr] = x, action, reward
                X2[ptr], D[ptr] = np.append(states[s + step_days, j], action), float(k == n - 1)
                ptr, filled, position = (ptr + 1) % size, min(filled + 1, size), action
                if filled >= p["batch"]:
                    idx = rng.integers(0, filled, p["batch"])
                    q = net.output(X[idx])
                    y = R[idx] + p["gamma"] * (1 - D[idx]) * target.output(X2[idx]).max(axis=1)
                    grad = np.zeros_like(q)
                    grad[np.arange(len(idx)), A[idx]] = q[np.arange(len(idx)), A[idx]] - y
                    net.update(X[idx], grad)
                    updates += 1
                    if updates % p["target_sync"] == 0:
                        target = net.clone()
        return net
