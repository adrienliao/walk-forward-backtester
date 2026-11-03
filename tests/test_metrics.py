import math

import numpy as np
import pandas as pd
import pytest

from btlab import metrics


def test_sharpe_constant_series_is_zero():
    assert metrics.sharpe(pd.Series([0.0] * 10)) == 0.0


def test_sharpe_scaling():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 1000))
    assert metrics.sharpe(r) == pytest.approx(metrics.sharpe(r, periods=1) * math.sqrt(252))
    assert metrics.sharpe(2 * r) == pytest.approx(metrics.sharpe(r))


def test_max_drawdown_known_path():
    r = pd.Series([0.10, -0.50, 0.20])
    # equity 1.1, 0.55, 0.66: worst point is 50% below the 1.1 peak
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_psr_rises_with_track_record_length():
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0.0005, 0.01, 250))
    long = pd.Series(rng.normal(0.0005, 0.01, 5000))
    assert metrics.probabilistic_sharpe(long) > metrics.probabilistic_sharpe(short)


def test_expected_max_sharpe_grows_with_trials():
    vals = [metrics.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_deflated_below_probabilistic():
    rng = np.random.default_rng(2)
    trials = [pd.Series(rng.normal(0, 0.01, 750)) for _ in range(200)]
    srs = [metrics.sharpe(t, periods=1) for t in trials]
    best = trials[int(np.argmax(srs))]
    assert metrics.deflated_sharpe(best, srs) < metrics.probabilistic_sharpe(best)
    assert metrics.deflated_sharpe(best, srs) < 0.9


def test_bootstrap_ci_contains_point_estimate():
    r = pd.Series(np.random.default_rng(3).normal(0.0004, 0.01, 2000))
    lo, hi = metrics.bootstrap_sharpe_ci(r, n_boot=500)
    assert lo < metrics.sharpe(r) < hi
