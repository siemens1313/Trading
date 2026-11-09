"""Strategy 15: genetic algorithm.

A population of trading rules evolves by tournament selection, uniform crossover, Gaussian mutation and
elitism. Each rule is a genome (fast and slow moving-average lengths, an RSI length and an RSI floor):
long a stock while its fast average is above its slow one and RSI is above the floor. Fitness is the
rule's Sharpe ratio after costs on the trailing `train_days`, minus a turnover penalty. The best genome
trades until the next evolution, `evolve_every` days later, so it is always judged out of sample.
"""

from __future__ import annotations

import math

import numpy as np

from ..indicators import rsi
from .base import DailyStrategy, first_fit

GENES = (("fast", 5, 50), ("slow", 20, 200), ("rsi_len", 2, 21), ("rsi_floor", 30.0, 70.0))


class _Indicators:
    """Moving averages and RSIs of any length, computed once and reused across genomes."""

    def __init__(self, close: np.ndarray):
        self.close = close
        self._cum = np.vstack([np.zeros((1, close.shape[1])), np.cumsum(close, axis=0)])
        self._rsi: dict[int, np.ndarray] = {}

    def sma(self, n: int) -> np.ndarray:
        out = np.full(self.close.shape, np.nan)
        out[n - 1 :] = (self._cum[n:] - self._cum[:-n]) / n
        return out

    def rsi(self, n: int) -> np.ndarray:
        if n not in self._rsi:
            self._rsi[n] = rsi(self.close, n)
        return self._rsi[n]

    def signal(self, genome: tuple) -> np.ndarray:
        fast, slow, rsi_len, floor = genome
        return (self.sma(int(fast)) > self.sma(int(slow))) & (self.rsi(int(rsi_len)) > floor)


class GeneticAlgorithm(DailyStrategy):
    name, number, title = "genetic", 15, "Genetic algorithm (evolved trend + RSI rule)"
    defaults = {
        "population": 24,
        "generations": 12,
        "evolve_every": 126,
        "train_days": 756,
        "min_train_days": 504,
        "mutation": 0.2,
        "crossover": 0.8,
        "elite": 2,
        "tournament": 3,
        "turnover_penalty": 0.002,
        "cost_bps": 5.0,
        "seed": 5,
    }

    def weights(self, panel):
        p = self.p
        T, N = panel.close.shape
        ind = _Indicators(panel.close)
        fwd = np.full((T, N), np.nan)
        fwd[: T - 2] = panel.open[2:] / panel.open[1 : T - 1] - 1
        W = np.zeros((T, N))
        genome = None
        self.history: list[tuple[str, tuple, float]] = []
        for t in range(T):
            if t % p["evolve_every"] == 0:
                lo, hi = self._window(t)
                if hi - lo >= p["min_train_days"]:
                    genome, fit, _ = self.evolve(ind, fwd, lo, hi, p["seed"] + t)
                    self.history.append((str(panel.dates[t]), genome, fit))
            if genome is not None:
                W[t] = ind.signal(genome)[t] / N
        return W

    def describe(self, panel):
        if not self.history:
            return ["not evolved yet: not enough history"]
        day, (fast, slow, rsi_len, floor), fit = self.history[-1]
        return [
            f"rule evolved {day}: long while the {fast}-day average is above the {slow}-day "
            f"and {rsi_len}-day RSI is above {floor} (in-sample fitness {fit:.2f})"
        ]

    def _window(self, t: int) -> tuple[int, int]:
        """Training rows at day t: s <= t-2, whose returns (open s+1 to open s+2) are known by the close of t."""
        return max(0, t - self.p["train_days"]), t - 1

    def warmup(self) -> int:
        return first_fit(self.p["evolve_every"], self.p["min_train_days"], self._window)

    def fitness(self, ind: _Indicators, fwd: np.ndarray, lo: int, hi: int, genome: tuple) -> float:
        held = ind.signal(genome)[lo:hi] / fwd.shape[1]
        turnover = np.abs(np.diff(held, axis=0, prepend=0)).sum(axis=1)
        r = (held * fwd[lo:hi]).sum(axis=1) - turnover * self.p["cost_bps"] / 1e4
        sd = r.std()
        if sd == 0:
            return -1.0
        return float(r.mean() / sd * math.sqrt(252) - self.p["turnover_penalty"] * turnover.sum() * 252 / len(r))

    def evolve(self, ind: _Indicators, fwd: np.ndarray, lo: int, hi: int, seed: int) -> tuple[tuple, float, list[float]]:
        """Best genome, its fitness, and the best fitness after each generation."""
        p = self.p
        rng = np.random.default_rng(seed)
        cache: dict[tuple, float] = {}

        def fit(g: tuple) -> float:
            if g not in cache:
                cache[g] = self.fitness(ind, fwd, lo, hi, g)
            return cache[g]

        pop = [self._repair(tuple(rng.uniform(a, b) for _, a, b in GENES)) for _ in range(p["population"])]
        best_by_generation = []
        for _ in range(p["generations"]):
            scores = [fit(g) for g in pop]
            order = np.argsort(scores)[::-1]
            best_by_generation.append(scores[order[0]])
            nxt = [pop[i] for i in order[: p["elite"]]]
            while len(nxt) < p["population"]:
                a, b = self._pick(pop, scores, rng), self._pick(pop, scores, rng)
                child = tuple(x if rng.random() < 0.5 else y for x, y in zip(a, b)) if rng.random() < p["crossover"] else a
                child = tuple(
                    v + rng.normal(0, 0.1 * (hi_ - lo_)) if rng.random() < p["mutation"] else v
                    for v, (_, lo_, hi_) in zip(child, GENES)
                )
                nxt.append(self._repair(child))
            pop = nxt
        scores = [fit(g) for g in pop]
        best = int(np.argmax(scores))
        best_by_generation.append(scores[best])
        return pop[best], scores[best], best_by_generation

    def _pick(self, pop: list, scores: list, rng) -> tuple:
        entrants = rng.integers(0, len(pop), self.p["tournament"])
        return pop[max(entrants, key=lambda i: scores[i])]

    @staticmethod
    def _repair(genome: tuple) -> tuple:
        """Clip genes to their ranges, round the lengths, and keep the fast average faster than the slow one."""
        fast, slow, rsi_len, floor = (min(max(v, lo), hi) for v, (_, lo, hi) in zip(genome, GENES))
        fast, slow = sorted((int(round(fast)), int(round(slow))))
        slow = max(slow, fast + 5)  # fast is at most 50, so this stays within the slow range
        return (fast, slow, int(round(rsi_len)), round(float(floor), 1))
