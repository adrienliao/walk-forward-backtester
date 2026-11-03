"""Performance statistics, including the ones that tell you when to distrust a Sharpe ratio.

The probabilistic and deflated Sharpe ratios follow Bailey and Lopez de Prado
(2012, 2014). The deflated version corrects for the fact that the best of N
backtests looks good even when none of them has any edge.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

ANN = 252
EULER_GAMMA = 0.5772156649015329


def ann_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (ANN / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * math.sqrt(ANN))


def sharpe(r: pd.Series, periods: int = ANN) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def sortino(r: pd.Series) -> float:
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * math.sqrt(ANN)) if downside > 0 else 0.0


def drawdown(r: pd.Series) -> pd.Series:
    equity = (1 + r).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar(r: pd.Series) -> float:
    mdd = max_drawdown(r)
    return ann_return(r) / abs(mdd) if mdd < 0 else float("nan")


def hit_rate(r: pd.Series) -> float:
    active = r[r != 0]
    return float((active > 0).mean()) if len(active) else float("nan")


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), accounting for sample length, skew and fat tails.

    Both Sharpe ratios are per period (not annualised) inside the formula.
    """
    n = len(r)
    sr = sharpe(r, periods=1)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, trial_sr_var: float) -> float:
    """Expected maximum of N Sharpe ratios drawn under the null of zero skill."""
    if n_trials < 2:
        return 0.0
    return math.sqrt(trial_sr_var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(r: pd.Series, trial_sharpes_per_period) -> float:
    """PSR against the Sharpe you would expect from the luckiest of the trials."""
    trial_sharpes_per_period = np.asarray(trial_sharpes_per_period, dtype=float)
    sr0 = expected_max_sharpe(len(trial_sharpes_per_period), trial_sharpes_per_period.var(ddof=1))
    return probabilistic_sharpe(r, sr_benchmark=sr0)


def bootstrap_sharpe_ci(r: pd.Series, block: int = 21, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Moving block bootstrap, which keeps the short range autocorrelation of returns."""
    rng = np.random.default_rng(seed)
    x = np.asarray(r, dtype=float)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    samples = x[idx]
    sr = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * math.sqrt(ANN)
    return float(np.quantile(sr, alpha / 2)), float(np.quantile(sr, 1 - alpha / 2))


def summary(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "hit_rate": hit_rate(r),
        "psr_vs_0": probabilistic_sharpe(r),
    }
    if turnover is not None:
        out["ann_turnover"] = float(turnover.mean() * ANN)
    return out
