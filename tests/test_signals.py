import numpy as np
import pandas as pd
import pytest

from btlab import signals


@pytest.mark.parametrize("name", list(signals.STRATEGIES))
def test_no_lookahead(noise, name):
    """Changing the future must not change any weight decided in the past."""
    fn, grid = signals.STRATEGIES[name]
    params = grid[len(grid) // 2]
    cut = 1000
    w_full = fn(noise, **params)
    tampered = noise.copy()
    tampered.iloc[cut:] *= np.random.default_rng(0).uniform(0.5, 1.5, size=tampered.iloc[cut:].shape)
    w_tampered = fn(tampered, **params)
    pd.testing.assert_frame_equal(w_full.iloc[:cut], w_tampered.iloc[:cut])


def test_xs_reversal_is_dollar_neutral(noise):
    w = signals.xs_reversal(noise, lookback=5).iloc[10:]
    assert np.allclose(w.sum(axis=1), 0.0)
    assert np.allclose(w.abs().sum(axis=1), 1.0)


def test_xs_reversal_buys_the_loser():
    idx = pd.bdate_range("2020-01-01", periods=3)
    px = pd.DataFrame({"up": [100, 110, 120], "down": [100, 90, 80], "flat": [100, 100, 100]}, index=idx)
    w = signals.xs_reversal(px, lookback=1).iloc[-1]
    assert w["down"] > 0 > w["up"]


def test_momentum_respects_leverage_cap(noise):
    w = signals.ts_momentum(noise, lookback=63, skip=0, target_vol=5.0, max_leverage=1.5)
    assert w.abs().sum(axis=1).max() <= 1.5 + 1e-12
