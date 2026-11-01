"""Walk-forward parameter selection.

For each fold the parameters with the best Sharpe on the trailing training
window are frozen and traded on the next test window. The test windows are
stitched together into a single out of sample track record. Because the
weights from every fold are stitched before running the engine, the cost of
switching parameters between folds is charged too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import engine, metrics


@dataclass
class WalkForwardResult:
    oos: engine.BacktestResult
    chosen: pd.DataFrame                   # one row per fold: dates and selected params
    full_sample: dict = field(repr=False)  # param label -> full sample BacktestResult
    oos_start: pd.Timestamp = None

    def best_in_sample_label(self) -> str:
        return max(self.full_sample, key=lambda k: metrics.sharpe(self.full_sample[k].net.loc[self.oos_start:]))


def _label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def walk_forward(prices: pd.DataFrame, strategy, grid: list[dict], train: int = 756, test: int = 126,
                 cost_bps: float = 1.0) -> WalkForwardResult:
    if len(prices) < train + test:
        raise ValueError("not enough data for a single fold")

    weights = {_label(p): strategy(prices, **p) for p in grid}
    full = {k: engine.run(prices, w, cost_bps) for k, w in weights.items()}

    stitched = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rows = []
    for start in range(train, len(prices), test):
        tr = slice(prices.index[start - train], prices.index[start - 1])
        te = prices.index[start:start + test]
        scores = {k: metrics.sharpe(res.net.loc[tr]) for k, res in full.items()}
        best = max(scores, key=scores.get)
        stitched.loc[te] = weights[best].loc[te]
        rows.append({"test_start": te[0], "test_end": te[-1], "params": best, "train_sharpe": scores[best]})

    oos_start = prices.index[train]
    oos = engine.run(prices, stitched, cost_bps)
    # drop the training period, nothing was traded there
    oos = engine.BacktestResult(*(getattr(oos, f).loc[oos_start:] for f in ("weights", "gross", "costs", "net", "turnover")))
    return WalkForwardResult(oos, pd.DataFrame(rows), full, oos_start)
