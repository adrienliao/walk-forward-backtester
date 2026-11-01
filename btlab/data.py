from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def load_prices(path: str | Path = DATA / "etf_close.csv") -> pd.DataFrame:
    """Daily adjusted closes of SPY and the nine original SPDR sector ETFs, 2004 to 2025."""
    return pd.read_csv(path, index_col="date", parse_dates=True)


def download(tickers, start="2004-01-01", end=None) -> pd.DataFrame:
    import yfinance as yf  # optional dependency, only needed to refresh the data

    px = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=True)["Close"]
    return px[list(tickers)].dropna()


def random_walks(n_days=5000, n_assets=9, vol=0.2, seed=0) -> pd.DataFrame:
    """GBM prices with zero drift and no predictability at all: the null hypothesis."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n_days, n_assets)) * vol / np.sqrt(252) - 0.5 * vol**2 / 252
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"A{i}" for i in range(n_assets)])
