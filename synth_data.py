import numpy as np
import pandas as pd

def make_synth(n=1000, seed=7):
    np.random.seed(seed)
    # Simulate log-returns with slight mean-reversion
    eps = np.random.randn(n)*0.01
    reversion = np.zeros(n)
    for t in range(1, n):
        reversion[t] = -0.2*eps[t-1]  # AR(1) negative coeff → mean reversion
    ret = eps + reversion
    price = 100*np.exp(np.cumsum(ret))
    df = pd.DataFrame({
        "price": price,
        "ret": np.r_[0, np.diff(np.log(price))]
    })
    return df

if __name__ == "__main__":
    df = make_synth()
    df.to_csv("data/synth.csv", index=False)
    print("Saved data/synth.csv")
