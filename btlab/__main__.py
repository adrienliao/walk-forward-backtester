"""python -m btlab xs_reversal --cost-bps 1"""
import argparse

import pandas as pd

from . import data, metrics, signals, walkforward


def report(name: str, cost_bps: float, train: int, test: int) -> pd.DataFrame:
    prices = data.load_prices()
    fn, grid = signals.STRATEGIES[name]
    wf = walkforward.walk_forward(prices[data.ETFS], fn, grid, train=train, test=test, cost_bps=cost_bps)

    best = wf.best_in_sample_label()
    best_is = wf.full_sample[best]
    trial_srs = [metrics.sharpe(r.net.loc[wf.oos_start:], periods=1) for r in wf.full_sample.values()]
    spy = prices["SPY"].pct_change().loc[wf.oos_start:]

    rows = {
        "walk-forward (out of sample)": metrics.summary(wf.oos.net, wf.oos.turnover),
        f"best in hindsight ({best})": {
            **metrics.summary(best_is.net.loc[wf.oos_start:], best_is.turnover.loc[wf.oos_start:]),
            "deflated_sharpe": metrics.deflated_sharpe(best_is.net.loc[wf.oos_start:], trial_srs),
        },
        "SPY buy and hold": metrics.summary(spy),
    }
    lo, hi = metrics.bootstrap_sharpe_ci(wf.oos.net)
    print(f"\n{name}, {len(grid)} parameter sets, {cost_bps} bp per unit of turnover, "
          f"out of sample from {wf.oos_start.date()}")
    print(f"walk-forward Sharpe 95% block bootstrap CI: [{lo:.2f}, {hi:.2f}]\n")
    table = pd.DataFrame(rows).T
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nparameters picked per fold:")
    print(wf.chosen["params"].value_counts().to_string())
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(prog="btlab")
    ap.add_argument("strategy", choices=list(signals.STRATEGIES))
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--train", type=int, default=756, help="training window in days")
    ap.add_argument("--test", type=int, default=126, help="test window in days")
    a = ap.parse_args(argv)
    report(a.strategy, a.cost_bps, a.train, a.test)


if __name__ == "__main__":
    main()
