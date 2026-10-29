# The EASE recommender (Steck, 2019).
# Learns how much liking one film predicts liking every other film.

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.linalg import blas, lapack

from filmmatch.paths import DATA_DIR

LIKE_THRESHOLD = 4.0  # stars that count as liking a film
WATCHED_WEIGHT = 0.5  # films watched but not liked
STAR_WEIGHTS = {0.5: 0.0, 1.0: 0.0, 1.5: 0.25, 2.0: 0.25, 3.5: 0.75}  # low ratings count less
LAM = 500.0  # how strongly the model is regularized
DROPOUT = 0.1  # EDLAE; 0.25 and 0.5 scored worse
ALPHA = 0.1  # popularity discount, see recommend()
MIN_RATINGS = 100  # films with fewer ratings are left out
MAX_FILMS = 10_000  # users who rated more films than this are dropped (bots)


def like_weights(ratings):
    """How much each row of ratings counts: liked (4+ stars or hearted) is 1, other
    watched films get WATCHED_WEIGHT, and STAR_WEIGHTS overrides that for specific ratings."""
    rating = ratings["rating"].to_numpy(float)
    hearted = ratings["liked"].fillna(False).to_numpy(bool)
    weight = np.where((rating >= LIKE_THRESHOLD) | hearted, 1.0, WATCHED_WEIGHT)
    for stars, w in STAR_WEIGHTS.items():
        weight[(rating == stars) & ~hearted] = w
    return weight.astype(np.float32)


def build_matrix(ratings, n_items):
    """Users x films matrix of like_weights. Users whose films all weigh 0 get no
    row. Returns (X, user_ids); row i of X belongs to user_ids[i]."""
    weight = like_weights(ratings)
    keep = weight != 0
    codes, users = pd.factorize(ratings["userId"])  # numbers are much faster than strings
    used, rows = np.unique(codes[keep], return_inverse=True)
    cols = ratings["idx"].to_numpy()[keep]
    X = sp.csr_matrix((weight[keep], (rows, cols)), shape=(len(used), n_items))
    X.sum_duplicates()
    return X, np.asarray(users)[used]


def user_vector(films, n_items):
    """One user's row from their films (idx, rating, liked), built the same way as build_matrix."""
    X, _ = build_matrix(films.assign(userId=0), n_items)
    return X


def gram(X, block=4096):
    """X'X, a block of users at a time. (X.T @ X).toarray() needs about 3x the memory."""
    n = X.shape[1]
    G = np.zeros((n, n), dtype=np.float32, order="F")
    for start in range(0, X.shape[0], block):
        Xb = np.asfortranarray(X[start : start + block].toarray())
        # G += Xb' Xb in place (upper triangle only)
        G = blas.ssyrk(1.0, Xb, beta=1.0, c=G, trans=1, overwrite_c=True)
    for start in range(0, n, block):  # copy the upper triangle into the lower
        stop = min(start + block, n)
        G[start:stop, :start] = G[:start, start:stop].T
        diag = G[start:stop, start:stop]
        low = np.tril_indices(stop - start, -1)
        diag[low] = diag.T[low]
    return G


def fit(X):
    """Train EASE and return B. DROPOUT > 0 is EDLAE (Steck, 2020), which holds
    popular films back more."""
    G = gram(X)
    G[np.diag_indices_from(G)] += LAM + DROPOUT / (1 - DROPOUT) * np.diag(G)
    # invert in place with Cholesky: one matrix in memory instead of three
    P, info = lapack.spotrf(G, lower=1, overwrite_a=1)
    assert info == 0, f"gram matrix not positive definite ({info})"
    P, info = lapack.spotri(P, lower=1, overwrite_c=1)
    assert info == 0, f"inverse failed ({info})"
    for i in range(len(P)):  # potri fills only the lower triangle
        P[i, i + 1 :] = P[i + 1 :, i]
    P /= -np.diag(P).copy()  # B[i, j] = -P[i, j] / P[j, j]
    np.fill_diagonal(P, 0.0)
    return P


def recommend(B, x, exclude, popularity, k=20):
    """Top k films for one user, skipping exclude. B can be read from disk.
    EASE favors popular films, so positive scores are divided by popularity**ALPHA."""
    scores = x.data @ np.asarray(B[x.indices], dtype=np.float32)  # == x @ B
    scores = np.where(scores > 0, scores / popularity**ALPHA, scores)
    scores[exclude] = -np.inf
    return np.argsort(-scores)[:k]


def load_data():
    """Load (films, logs) from scripts/recommender/build.py. Keeps films with MIN_RATINGS or
    more ratings, and drops users with over MAX_FILMS rated films (bots)."""
    films = pd.read_parquet(DATA_DIR / "films.parquet")
    films = films[films["n_rated"] >= MIN_RATINGS].reset_index(drop=True)
    films["idx"] = np.arange(len(films))

    # read as categories: 90M strings would need about 10 GB
    logs = pq.read_table(DATA_DIR / "ratings.parquet", read_dictionary=["user", "slug"]).to_pandas()
    n_rated = logs.dropna(subset=["rating"]).groupby("user", observed=True).size()
    logs = logs[logs["user"].isin(n_rated.index[n_rated <= MAX_FILMS])]
    idx = dict(zip(films["slug"], films["idx"]))
    logs = logs[logs["slug"].isin(idx)]
    logs["idx"] = logs["slug"].map(idx).astype(np.int32)
    return films, logs.rename(columns={"user": "userId"})[["userId", "idx", "rating", "liked"]]
