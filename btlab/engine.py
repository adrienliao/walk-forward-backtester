"""Vectorised backtest engine for daily target weight strategies.

Convention: a strategy returns target weights computed with information up to
and including the close of day t. They are applied to the return from close t
to close t+1, which is why the engine lags them by one bar. Costs are charged
on the absolute change in weights (turnover), so flipping from -1 to +1 costs
twice as much as opening a position.

Weights are assumed to be rebalanced back to target every day, so drift
between rebalances is ignored. That is fine for daily strategies and keeps
the engine a few lines long.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    weights: pd.DataFrame   # weights actually held over each day's return
    gross: pd.Series
    costs: pd.Series
    net: pd.Series
    turnover: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()


def run(prices: pd.DataFrame, target_weights: pd.DataFrame, cost_bps: float = 1.0, lag: int = 1) -> BacktestResult:
    if lag < 1:
        raise ValueError("lag must be at least 1, otherwise the strategy trades on information it does not have yet")
    target_weights = target_weights.reindex_like(prices)
    rets = prices.pct_change().fillna(0.0)
    held = target_weights.shift(lag).fillna(0.0)

    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    costs = turnover * cost_bps / 1e4
    return BacktestResult(held, gross, costs, gross - costs, turnover)


def buy_and_hold(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)


def normalise_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    total = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(total, axis=0).mul(gross).fillna(0.0)
