"""Covariance estimation: EWMA volatilities x Ledoit-Wolf-shrunk correlations.

Sample correlation matrices from ~6 months of data are noisy, and an optimizer will happily exploit
that noise. Shrinking toward the identity is the standard fix (Ledoit & Wolf, 2004)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrunk covariance of the rows of X (n_obs x p) toward mu*I. Returns (cov, shrinkage)."""
    n, p = X.shape
    X = X - X.mean(axis=0)
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    X2 = X ** 2
    beta = ((X2.T @ X2) / n - S ** 2).sum() / (p * n)
    shrink = 0.0 if delta == 0 else float(min(beta, delta) / delta)
    return shrink * mu * np.eye(p) + (1 - shrink) * S, shrink


def ewma_vol(rets: pd.DataFrame, halflife: int) -> pd.DataFrame:
    return np.sqrt((rets ** 2).ewm(halflife=halflife, min_periods=halflife).mean())


def covariance(rets: pd.DataFrame, vol_now: pd.Series, corr_window: int) -> np.ndarray:
    """Daily covariance matrix as of the last row of `rets` (columns in order)."""
    R = rets.iloc[-corr_window:].dropna(axis=0, how="any")
    if len(R) < max(20, corr_window // 3):
        raise ValueError("not enough clean return history for the risk model")
    sd = R.std().replace(0, np.nan)
    Z = (R / sd).fillna(0.0).values
    corr, _ = ledoit_wolf(Z)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    v = vol_now.values
    return corr * np.outer(v, v)
