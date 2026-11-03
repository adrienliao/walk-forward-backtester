import numpy as np
import pandas as pd
import pytest

from btlab import engine


def _prices(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=idx, dtype=float)


def test_weights_are_applied_to_the_next_return():
    px = _prices([100, 110, 99])
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=px.index)
    res = engine.run(px, w, cost_bps=0)
    # decided at close of day 0, earns the day 0 -> day 1 move only
    assert res.net.tolist() == pytest.approx([0.0, 0.10, 0.0])


def test_costs_scale_with_turnover():
    px = _prices([100] * 4)
    w = pd.DataFrame({"A": [1.0, -1.0, -1.0, 0.0]}, index=px.index)
    res = engine.run(px, w, cost_bps=10)
    assert res.turnover.tolist() == pytest.approx([0.0, 1.0, 2.0, 0.0])
    assert res.costs.sum() == pytest.approx(3 * 10 / 1e4)


def test_zero_lag_is_refused():
    px = _prices([100, 101])
    with pytest.raises(ValueError):
        engine.run(px, px * 0, lag=0)


def test_normalise_gross():
    w = pd.DataFrame({"A": [2.0, 0.0], "B": [-2.0, 0.0]})
    out = engine.normalise_gross(w)
    assert out.abs().sum(axis=1).tolist() == [1.0, 0.0]
