# Test the recommender on users it never saw.
# Hide 20% of each test user's films and see how many of the ones they liked come back in the top k.

import numpy as np
import pandas as pd
import scipy.sparse as sp

from filmmatch.recommender import LIKE_THRESHOLD, like_weights


def split_users(ratings, n_test, seed=0):
    """Split the ratings table into (train, test) by user."""
    users = ratings["userId"].unique()
    test_users = np.random.default_rng(seed).choice(users, n_test, replace=False)
    is_test = ratings["userId"].isin(test_users)
    return ratings[~is_test], ratings[is_test]


def hide(test, share=0.2, seed=0):
    """Split held-out users' logs into (shown, hidden), hiding each film with probability share."""
    hidden = np.random.default_rng(seed).random(len(test)) < share
    return test[~hidden], test[hidden]


def rows(df, users, n_items, values=None):
    """users x films matrix with values (default 1) at each (user, film) in df."""
    values = np.ones(len(df), np.float32) if values is None else values
    codes = pd.Categorical(df["userId"], categories=users).codes
    return sp.csr_matrix((values, (codes, df["idx"])), shape=(len(users), n_items))


def holdout(B, shown, hidden, n_items, k=20, chunk=1000):
    """Per-user recall, ndcg, and how many hidden films in the top k were bad or rated."""
    users = np.unique(np.concatenate([shown["userId"], hidden["userId"]]))
    X = rows(shown, users, n_items, like_weights(shown))
    seen = rows(shown, users, n_items)
    liked = (hidden["rating"] >= LIKE_THRESHOLD) | hidden["liked"]
    like = rows(hidden[liked], users, n_items)
    rated = rows(hidden[hidden["rating"].notna()], users, n_items)
    bad = rows(hidden[hidden["rating"] <= 2], users, n_items)
    n_like = np.asarray(like.sum(axis=1)).ravel()
    discount = 1 / np.log2(np.arange(k) + 2)  # ndcg: hits near the top count more
    ideal = np.concatenate([[0], np.cumsum(discount)])[np.minimum(n_like, k).astype(int)]

    out = {m: np.empty(len(users)) for m in ("recall", "ndcg", "bad", "rated")}
    for start in range(0, len(users), chunk):
        part = slice(start, start + chunk)
        scores = X[part] @ B
        scores[seen[part].toarray() > 0] = -np.inf
        top = np.argsort(-scores, axis=1)[:, :k]

        def in_top(m):  # which of the top k are in m, per user
            return np.take_along_axis(m[part].toarray() > 0, top, axis=1)

        hits = in_top(like)
        n = n_like[part]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["recall"][part] = np.where(n > 0, hits.sum(axis=1) / np.minimum(n, k), np.nan)
            out["ndcg"][part] = np.where(n > 0, (hits * discount).sum(axis=1) / ideal[part], np.nan)
        out["bad"][part] = in_top(bad).sum(axis=1)
        out["rated"][part] = in_top(rated).sum(axis=1)
    return out
