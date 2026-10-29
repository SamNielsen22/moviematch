# Test the recommender on users it never saw.
# Hide 20% of each test user's films and see how many of the ones they liked come back in the top k.

import numpy as np
import pandas as pd
import scipy.sparse as sp

from filmmatch.recommender import ALPHA, LIKE_THRESHOLD, like_weights


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
    values = np.ones(len(df), np.float32) if values is None else np.asarray(values, np.float32)
    codes = pd.Categorical(df["userId"], categories=users).codes
    X = sp.csr_matrix((values, (codes, df["idx"].to_numpy())), shape=(len(users), n_items))
    X.eliminate_zeros()
    X.sum_duplicates()
    return X


def holdout(B, shown, hidden, n_items, popularity, k=20, chunk=1000):
    """Per-user recall, ndcg, and how many hidden films in the top k were bad or rated."""
    users = np.sort(pd.concat([shown["userId"], hidden["userId"]]).astype(str).unique())
    X = rows(shown, users, n_items, like_weights(shown))
    seen = rows(shown, users, n_items)
    liked = (hidden["rating"] >= LIKE_THRESHOLD) | hidden["liked"].fillna(False).astype(bool)
    like = rows(hidden[liked], users, n_items)
    rated = rows(hidden[hidden["rating"].notna()], users, n_items)
    bad = rows(hidden[hidden["rating"] <= 2], users, n_items)
    n_like = np.asarray(like.sum(axis=1)).ravel()
    discount = 1 / np.log2(np.arange(k) + 2)
    ideal = np.concatenate([[0], np.cumsum(discount)])[np.minimum(n_like, k).astype(int)]

    out = {m: np.empty(len(users)) for m in ("recall", "ndcg", "bad", "rated")}
    for start in range(0, len(users), chunk):
        part = slice(start, start + chunk)
        s = np.asarray(X[part] @ B, dtype=np.float32)
        s[seen[part].toarray() > 0] = -np.inf
        s = np.where(s > 0, s / popularity**ALPHA, s)  # same popularity discount as recommend()
        top = np.argpartition(-s, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, axis=1), axis=1), axis=1)
        hit = {name: np.take_along_axis(m[part].toarray() > 0, top, axis=1)
               for name, m in (("like", like), ("rated", rated), ("bad", bad))}
        n = n_like[part]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["recall"][part] = np.where(n > 0, hit["like"].sum(axis=1) / np.minimum(n, k), np.nan)
            out["ndcg"][part] = np.where(n > 0, (hit["like"] * discount).sum(axis=1) / ideal[part], np.nan)
        out["bad"][part] = hit["bad"].sum(axis=1)
        out["rated"][part] = hit["rated"].sum(axis=1)
    return out

