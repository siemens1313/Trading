"""Cost-aware mean-variance portfolio construction, dependency-free.

    maximize   mu'w  -  (lambda*h/2) w'Sigma w  -  kappa * sum_i c_i |w_i - w0_i|
    subject to lo_i <= w_i <= hi_i

solved exactly by FISTA (accelerated proximal gradient). The prox of the cost term plus box is
closed-form: soft-threshold toward the current weight w0, then clip. The L1 cost creates a no-trade
band, so the book only trades when the forecast improvement beats the cost of trading.
Gross exposure and volatility caps are applied afterwards by scaling down, never up."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptResult:
    w: np.ndarray
    iters: int
    ex_ante_vol: float      # annualized
    scaled_by: float


def _prox(v, w0, thresh, lo, hi):
    # soft-threshold toward w0, then clip to the box (exact for a separable 1-D convex prox)
    x = np.maximum(v - thresh, np.minimum(v + thresh, w0))
    return np.minimum(np.maximum(x, lo), hi)


def _fista(mu, Q, w0, cost_thresh_unit, lo, hi, w_init, max_iter, tol):
    L = float(np.linalg.eigvalsh(Q).max())
    step = 1.0 / max(L, 1e-12)
    thresh = step * cost_thresh_unit
    w = np.clip(w_init, lo, hi)
    y, t, it = w.copy(), 1.0, 0
    for it in range(1, max_iter + 1):
        w_new = _prox(y - step * (Q @ y - mu), w0, thresh, lo, hi)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = w_new + ((t - 1) / t_new) * (w_new - w)
        done = np.abs(w_new - w).max() < tol
        w, t = w_new, t_new
        if done:
            break
    return w, it


def solve(mu: np.ndarray, cov_daily: np.ndarray, w0: np.ndarray, cost: np.ndarray, *,
          horizon: int, risk_aversion: float, tcost_aversion: float, lo: np.ndarray, hi: np.ndarray,
          max_gross: float, target_vol: float, calibrate: bool = True,
          max_iter: int = 300, tol: float = 1e-7) -> OptResult:
    """If calibrate=True, risk aversion is re-solved (a few Newton-like steps on vol ~ 1/lambda) so the
    portfolio uses its volatility budget, like a PM handed a risk limit. Caps are then enforced by
    scaling down only."""
    mu = np.nan_to_num(mu)
    lam, w, iters = risk_aversion, np.clip(w0, lo, hi), 0
    ann = lambda x: float(np.sqrt(max(x @ cov_daily @ x, 0.0) * 252))  # noqa: E731
    for _ in range(4 if calibrate else 1):
        w, it = _fista(mu, lam * horizon * cov_daily, w0, tcost_aversion * cost, lo, hi, w, max_iter, tol)
        iters += it
        vol, gross = ann(w), np.abs(w).sum()
        if not calibrate or vol <= 0:
            break
        if gross >= max_gross * 0.999 and vol <= target_vol:
            break  # gross-limited, not risk-limited
        ratio = vol / target_vol
        if abs(ratio - 1) < 0.05:
            break
        lam = float(np.clip(lam * ratio, 1e-3, 1e4))
    scale = 1.0
    gross = np.abs(w).sum()
    if gross > max_gross:
        scale = max_gross / gross
    if ann(w) * scale > target_vol:
        scale *= target_vol / (ann(w) * scale)
    w = w * scale
    return OptResult(w=w, iters=iters, ex_ante_vol=ann(w), scaled_by=scale)


def objective(w, mu, cov_daily, w0, cost, horizon, risk_aversion, tcost_aversion):
    return float(mu @ w - 0.5 * risk_aversion * horizon * w @ cov_daily @ w
                 - tcost_aversion * np.sum(cost * np.abs(w - w0)))
