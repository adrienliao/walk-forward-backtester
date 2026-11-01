"""Regenerate every figure in docs/. Run from the repo root: python scripts/make_figures.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btlab import data, engine, metrics, signals, walkforward  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
COLORS = {"ts_mean_reversion": BLUE, "xs_reversal": ORANGE, "ts_momentum": AQUA}
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})
OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)

prices = data.load_prices()
sectors = prices[data.ETFS]


def equity_curves(cost_bps=1.0):
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    start = None
    for name, (fn, grid) in signals.STRATEGIES.items():
        wf = walkforward.walk_forward(sectors, fn, grid, cost_bps=cost_bps)
        start = wf.oos_start
        r = wf.oos.net
        ax.plot((1 + r).cumprod(), color=COLORS[name], lw=1.6, label=f"{name}  (Sharpe {metrics.sharpe(r):.2f})")
        ax_dd.plot(metrics.drawdown(r), color=COLORS[name], lw=1.2)
    spy = prices["SPY"].pct_change().loc[start:].fillna(0)
    ax.plot((1 + spy).cumprod(), color=GREY, lw=1.2, ls="--", label=f"SPY  (Sharpe {metrics.sharpe(spy):.2f})")
    ax_dd.plot(metrics.drawdown(spy), color=GREY, lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("growth of 1 (log scale)")
    ax.set_title(f"Walk-forward out of sample, net of {cost_bps:g} bp costs", loc="left")
    ax.legend(loc="upper left")
    ax_dd.set_ylabel("drawdown")
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(OUT / "equity_curves.png", dpi=160)


def cost_sensitivity():
    costs = [0, 0.5, 1, 2, 3, 5]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (fn, grid) in signals.STRATEGIES.items():
        srs = [metrics.sharpe(walkforward.walk_forward(sectors, fn, grid, cost_bps=c).oos.net) for c in costs]
        ax.plot(costs, srs, "o-", color=COLORS[name], lw=2, ms=5, label=name)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("transaction cost, bp per unit of turnover")
    ax.set_ylabel("out of sample Sharpe")
    ax.set_title("High turnover edges disappear fast once you pay to trade", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cost_sensitivity.png", dpi=160)


def overfitting_on_noise(n_trials=300, seed=7):
    """Search hundreds of parameter sets on pure noise and see what the best one looks like."""
    rng = np.random.default_rng(seed)
    noise = data.random_walks(n_days=756, seed=seed)
    results = []
    for _ in range(n_trials):
        if rng.random() < 0.5:
            w = signals.ts_mean_reversion(noise, window=int(rng.integers(5, 80)), entry_z=float(rng.uniform(0.25, 2.5)))
        else:
            w = signals.ts_momentum(noise, lookback=int(rng.integers(10, 200)), skip=int(rng.integers(0, 10)), vol_window=20)
        results.append(engine.run(noise, w, cost_bps=0).net.iloc[200:])
    srs = np.array([metrics.sharpe(r) for r in results])
    best = int(np.argmax(srs))
    per_period = [metrics.sharpe(r, periods=1) for r in results]
    psr = metrics.probabilistic_sharpe(results[best])
    dsr = metrics.deflated_sharpe(results[best], per_period)
    sr0 = metrics.expected_max_sharpe(n_trials, np.var(per_period, ddof=1)) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(srs, bins=30, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.axvline(srs[best], color=ORANGE, lw=2)
    ax.axvline(sr0, color=GREY, lw=1.2, ls="--")
    ax.text(0.02, 0.97,
            f"orange: best of {n_trials} trials, Sharpe {srs[best]:.2f}\n"
            f"probabilistic Sharpe {psr:.2f} (looks significant)\n"
            f"deflated Sharpe {dsr:.2f} (it is not)\n"
            f"dashed: expected best under the null, {sr0:.2f}",
            transform=ax.transAxes, va="top", ha="left", color=GREY, fontsize=9)
    ax.set_xlabel("annualised Sharpe ratio")
    ax.set_ylabel("number of trials")
    ax.set_title(f"{n_trials} backtests on random walks: the winner has no edge", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "overfitting_on_noise.png", dpi=160)
    return srs[best], psr, dsr


if __name__ == "__main__":
    equity_curves()
    cost_sensitivity()
    print("noise experiment (best SR, PSR, DSR):", overfitting_on_noise())
    print("figures written to", OUT)
