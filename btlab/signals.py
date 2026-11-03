"""Strategies as functions: prices in, target weights out.

Each function only looks backwards (rolling windows and past returns), which
is checked by a test that truncates the future and verifies nothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import normalise_gross


def ts_mean_reversion(prices: pd.DataFrame, window: int = 20, entry_z: float = 1.0) -> pd.DataFrame:
    """Fade large deviations of log price from its rolling mean, asset by asset."""
    logp = np.log(prices)
    mean = logp.rolling(window).mean()
    std = logp.rolling(window).std(ddof=0)
    z = (logp - mean) / std
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[z > entry_z] = -1.0
    w[z < -entry_z] = 1.0
    return w / prices.shape[1]


def xs_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Cross sectional short term reversal: long recent losers, short recent winners.

    Weights are proportional to the demeaned rank, so the book is dollar neutral
    with gross exposure 1.
    """
    past = prices.pct_change(lookback)
    ranks = past.rank(axis=1)
    w = -(ranks.sub(ranks.mean(axis=1), axis=0))
    return normalise_gross(w)


def ts_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21, vol_window: int = 63,
                target_vol: float = 0.10, max_leverage: float = 2.0) -> pd.DataFrame:
    """Time series momentum (Moskowitz, Ooi, Pedersen 2012) with volatility targeting.

    Sign of the return from t-lookback to t-skip, each asset scaled to the same
    ex ante volatility, then the whole book capped at `max_leverage` gross.
    """
    signal = np.sign(prices.shift(skip) / prices.shift(lookback) - 1.0)
    vol = prices.pct_change().rolling(vol_window).std() * np.sqrt(252)
    w = signal * (target_vol / vol) / prices.shape[1]
    gross = w.abs().sum(axis=1)
    scale = (max_leverage / gross).clip(upper=1.0)
    return w.mul(scale, axis=0).fillna(0.0)


STRATEGIES = {
    "ts_mean_reversion": (ts_mean_reversion, [{"window": w, "entry_z": z} for w in (10, 20, 40, 60) for z in (0.5, 1.0, 1.5, 2.0)]),
    "xs_reversal": (xs_reversal, [{"lookback": k} for k in (1, 2, 3, 5, 10, 21)]),
    "ts_momentum": (ts_momentum, [{"lookback": k, "skip": s} for k in (63, 126, 252) for s in (0, 21)]),
}
