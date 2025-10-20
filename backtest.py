import argparse
import numpy as np
import pandas as pd

def zscore(x, window=50):
    roll = x.rolling(window)
    return (x - roll.mean())/roll.std(ddof=0)

def sharpe(returns, freq=252):
    mu = returns.mean()*freq
    sd = returns.std(ddof=0)*np.sqrt(freq)
    return mu/sd if sd>0 else 0.0

def max_drawdown(cum):
    peaks = cum.cummax()
    dd = (cum/peaks)-1.0
    return dd.min()

def run(z=1.0, cost_bps=1.0, window=50, test_frac=0.3, path="data/synth.csv"):
    df = pd.read_csv(path)
    df["z"] = zscore(df["ret"], window=window)
    df["signal"] = 0.0
    df.loc[df["z"] > z, "signal"] = -1.0  # mean reversion: short if high
    df.loc[df["z"] < -z, "signal"] = 1.0  # long if low

    # Positions are previous signal (enter next bar)
    df["position"] = df["signal"].shift(1).fillna(0.0)

    # Transaction costs on signal changes
    turns = (df["position"] != df["position"].shift(1)).astype(float)
    cost = turns * (cost_bps/10000.0)

    strat_ret = df["position"]*df["ret"] - cost
    df["cum"] = (1.0 + strat_ret).cumprod()

    # Time split
    n = len(df)
    split = int(n*(1-test_frac))
    train, test = df.iloc[:split], df.iloc[split:]

    metrics = {
        "train_sharpe": sharpe(train["position"]*train["ret"] - cost.iloc[:split]),
        "test_sharpe": sharpe(test["position"]*test["ret"] - cost.iloc[split:]),
        "test_max_drawdown": max_drawdown(test["cum"]),
        "final_cum_return": df["cum"].iloc[-1]-1.0
    }
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--cost_bps", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    from synth_data import make_synth
    df = make_synth()
    df.to_csv("data/synth.csv", index=False)
    metrics = run(z=args.z, cost_bps=args.cost_bps, window=args.window)
    print(metrics)
