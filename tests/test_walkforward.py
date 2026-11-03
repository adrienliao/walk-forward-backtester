import numpy as np
import pytest

from btlab import metrics, signals, walkforward


def test_folds_cover_the_test_period(noise):
    fn, grid = signals.STRATEGIES["xs_reversal"]
    wf = walkforward.walk_forward(noise, fn, grid, train=500, test=100)
    assert wf.oos.net.index[0] == noise.index[500]
    assert wf.oos.net.index[-1] == noise.index[-1]
    assert len(wf.chosen) == int(np.ceil((len(noise) - 500) / 100))


def test_selection_only_uses_the_past(noise):
    """Wiping out the data after a fold must not change which parameters that fold picked."""
    fn, grid = signals.STRATEGIES["ts_mean_reversion"]
    full = walkforward.walk_forward(noise, fn, grid, train=500, test=100)
    short = walkforward.walk_forward(noise.iloc[:900], fn, grid, train=500, test=100)
    assert full.chosen["params"].iloc[:4].tolist() == short.chosen["params"].iloc[:4].tolist()


def test_no_edge_on_noise(noise):
    fn, grid = signals.STRATEGIES["ts_momentum"]
    wf = walkforward.walk_forward(noise, fn, grid, train=500, test=100, cost_bps=0)
    assert abs(metrics.sharpe(wf.oos.net)) < 1.0


def test_too_short_raises(noise):
    fn, grid = signals.STRATEGIES["xs_reversal"]
    with pytest.raises(ValueError):
        walkforward.walk_forward(noise.iloc[:100], fn, grid, train=500, test=100)
