# Mini Backtest — Mean Reversion (Interview-sized)

Minimal backtesting scaffold for a **z-score mean reversion** signal on synthetic OHLCV data.

## Features
- Signal: z-score of returns; long/short when |z| > threshold
- PnL, cumulative return, **Sharpe**, **max drawdown**
- Walk-forward style split (train/test by time)
- Transaction costs & slippage
- Reproducible seed and simple data

## Quickstart
```bash
pip install -r requirements.txt
python synth_data.py
python backtest.py --cost_bps 1.0 --z 1.0
```
