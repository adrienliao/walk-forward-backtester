from backtest import sharpe, max_drawdown
import pandas as pd

def test_sharpe_zero_vol():
    s = pd.Series([0,0,0,0], dtype=float)
    assert sharpe(s) == 0.0

def test_drawdown_basics():
    import pandas as pd
    cum = pd.Series([1.0, 1.1, 1.05, 1.2, 1.0])
    dd = max_drawdown(cum)
    assert dd < 0
