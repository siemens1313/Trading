"""Small learning models in plain numpy for the machine-learning and reinforcement-learning strategies.

They're deliberately simple, seeded so every run is reproducible, and trained only on data the calling
strategy is allowed to see.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class Standardizer:
    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class MLP:
    """Dense network: ReLU hidden layers, a linear output layer, trained with Adam."""

    def __init__(self, sizes: list[int], seed: int = 0, lr: float = 1e-3, l2: float = 1e-4):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.lr, self.l2, self.steps = lr, l2, 0
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        acts = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if k < len(self.W) - 1 else z)
        return acts

    def output(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float))[-1]

    def update(self, X: np.ndarray, grad_out: np.ndarray) -> None:
        """One Adam step, given dLoss/dOutput for each row of the batch."""
        acts = self._forward(X)
        g = grad_out / len(X)
        grads_w, grads_b = [None] * len(self.W), [None] * len(self.W)
        for k in reversed(range(len(self.W))):
            grads_w[k] = acts[k].T @ g + self.l2 * self.W[k]
            grads_b[k] = g.sum(axis=0)
            if k:
                g = (g @ self.W[k].T) * (acts[k] > 0)
        self.steps += 1
        for i, (param, grad) in enumerate(zip(self.W + self.b, grads_w + grads_b)):
            self._m[i] = 0.9 * self._m[i] + 0.1 * grad
            self._v[i] = 0.999 * self._v[i] + 0.001 * grad * grad
            m_hat = self._m[i] / (1 - 0.9**self.steps)
            v_hat = self._v[i] / (1 - 0.999**self.steps)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def clone(self) -> MLP:
        """A frozen copy of the weights (the target network in DQN)."""
        other = MLP.__new__(MLP)
        other.W = [w.copy() for w in self.W]
        other.b = [b.copy() for b in self.b]
        other.lr, other.l2, other.steps = self.lr, self.l2, 0
        other._m = [np.zeros_like(p) for p in other.W + other.b]
        other._v = [np.zeros_like(p) for p in other.W + other.b]
        return other

    def fit_binary(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, batch: int = 128, seed: int = 0) -> MLP:
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch):
                idx = order[start : start + batch]
                p = sigmoid(self.output(X[idx])[:, 0])
                self.update(X[idx], (p - y[idx])[:, None])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.output(X)[:, 0])


class RandomForest:
    """Bagged shallow decision trees; Gini splits tried at a few quantile thresholds per feature."""

    def __init__(self, n_trees: int = 25, max_depth: int = 4, min_leaf: int = 40, n_thresholds: int = 12, seed: int = 0):
        self.n_trees, self.max_depth, self.min_leaf, self.n_thresholds, self.seed = n_trees, max_depth, min_leaf, n_thresholds, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        rng = np.random.default_rng(self.seed)
        n, f = X.shape
        m = max(1, int(round(np.sqrt(f))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)
            self.trees.append(self._grow(X[idx], y[idx], 0, rng, m))
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int, rng, m: int):
        p = float(y.mean())
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or p in (0.0, 1.0):
            return p
        best = None
        for f in rng.choice(X.shape[1], m, replace=False):
            col = X[:, f]
            for th in np.unique(np.quantile(col, np.linspace(0.1, 0.9, self.n_thresholds))):
                left = col <= th
                nl = int(left.sum())
                nr = len(y) - nl
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                pl, pr = y[left].mean(), y[~left].mean()
                impurity = nl * pl * (1 - pl) + nr * pr * (1 - pr)
                if best is None or impurity < best[0]:
                    best = (impurity, f, th, left)
        if best is None:
            return p
        _, f, th, left = best
        return (f, th, self._grow(X[left], y[left], depth + 1, rng, m), self._grow(X[~left], y[~left], depth + 1, rng, m))

    @classmethod
    def _predict(cls, node, X: np.ndarray) -> np.ndarray:
        if not isinstance(node, tuple):
            return np.full(len(X), node)
        f, th, left, right = node
        mask = X[:, f] <= th
        out = np.empty(len(X))
        out[mask] = cls._predict(left, X[mask])
        out[~mask] = cls._predict(right, X[~mask])
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([self._predict(tree, X) for tree in self.trees], axis=0)


class LinearSVM:
    """Linear support-vector machine (mini-batch Pegasos), with Platt scaling so its scores read as probabilities."""

    def __init__(self, lam: float = 1e-3, epochs: int = 10, batch: int = 64, seed: int = 0):
        self.lam, self.epochs, self.batch, self.seed = lam, epochs, batch, seed

    def fit(self, X: np.ndarray, y01: np.ndarray) -> LinearSVM:
        rng = np.random.default_rng(self.seed)
        y = 2 * y01 - 1
        Xb = np.hstack([X, np.ones((len(X), 1))])
        w = np.zeros(Xb.shape[1])
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch):
                idx = order[start : start + self.batch]
                step += 1
                eta = 1 / (self.lam * step)
                violated = y[idx] * (Xb[idx] @ w) < 1
                w = (1 - eta * self.lam) * w + eta / len(idx) * (y[idx][violated, None] * Xb[idx][violated]).sum(axis=0)
        self.w = w
        self._platt(self.decision(X), y01)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w[:-1] + self.w[-1]

    def _platt(self, d: np.ndarray, y01: np.ndarray, iters: int = 300, lr: float = 0.5) -> None:
        a, b = 1.0, 0.0
        for _ in range(iters):
            err = sigmoid(a * d + b) - y01
            a -= lr * float(np.mean(err * d))
            b -= lr * float(np.mean(err))
        self.a, self.b = a, b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * self.decision(X) + self.b)
