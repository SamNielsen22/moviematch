# Test the recommender with the settings fit.py uses.
# Needs as much memory as fit.py.

import numpy as np

from filmmatch.evaluate import hide, holdout, split_users
from filmmatch.recommender import build_matrix, fit, load_data

N_TEST = 10_000  # users held out
K = 20


def interval(x, w=None, n=1000):
    """Mean of x (or sum x / sum w) with a 95% bootstrap interval over users."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    ok = ~np.isnan(x)
    x, w = x[ok], w[ok]
    pick = np.random.default_rng(0).integers(0, len(x), (n, len(x)))
    means = x[pick].sum(axis=1) / w[pick].sum(axis=1)
    return f"{x.sum() / w.sum():.4f} [{np.percentile(means, 2.5):.4f}, {np.percentile(means, 97.5):.4f}]"


def main():
    films, logs = load_data()
    train, test = split_users(logs, N_TEST)
    shown, hidden = hide(test)
    popularity = films["n_rated"].to_numpy(np.float32)
    print(f"{len(films)} films, {logs['userId'].nunique()} users ({N_TEST} held out)", flush=True)

    X, _ = build_matrix(train, len(films))
    B = fit(X)
    scores = holdout(B, shown, hidden, len(films), popularity, K)
    print("recall", interval(scores["recall"]))
    print("ndcg", interval(scores["ndcg"]))
    print("bad", interval(scores["bad"], scores["rated"]))


if __name__ == "__main__":
    main()
