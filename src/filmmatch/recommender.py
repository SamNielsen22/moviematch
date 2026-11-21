# The EASE recommender (Steck, 2019).
# Learns how much liking one film predicts liking every other film.

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.linalg import blas, lapack

from filmmatch.paths import DATA_DIR

LIKE_THRESHOLD = 4.0  # stars that count as liking a film
WATCHED_WEIGHT = 0.5  # films logged without a rating
STAR_WEIGHTS = {0.5: 0, 1: 0, 1.5: 0.25, 2: 0.25, 2.5: 0.5, 3: 0.5, 3.5: 0.75, 4: 1, 4.5: 1, 5: 1}  # low ratings count less
LAM = 500.0  # how strongly the model is regularized
MIN_RATINGS = 100  # films with fewer ratings are left out
MAX_FILMS = 10_000  # users who rated more films than this are dropped (bots)
USERS_PER_BLOCK = 4096  # users gram() turns dense at a time, about 1 GB


def like_weights(ratings):
    """Weight each rating by how much it counts as liking the film."""
    weights = ratings["rating"].map(STAR_WEIGHTS)
    weights = weights.fillna(WATCHED_WEIGHT)
    weights[ratings["liked"]] = 1.0
    return weights.to_numpy(np.float32)


def build_matrix(ratings, n_items):
    """Build a sparse matrix of users x films, weighted by how much they liked each film."""
    weights = like_weights(ratings)
    rows = ratings["userId"]
    films = ratings["idx"]
    X = sp.csr_matrix((weights, (rows, films)), shape=(rows.max() + 1, n_items))
    return X


def gram(X):
    """Return X.T @ X with only the lower triangle filled in."""
    n_films = X.shape[1]
    G = np.zeros((n_films, n_films), dtype=np.float32, order="F")
    
    for start in range(0, X.shape[0], USERS_PER_BLOCK):
        Xb = np.asfortranarray(X[start : start + USERS_PER_BLOCK].toarray())
        # G += Xb' Xb in place (lower triangle only)
        G = blas.ssyrk(1.0, Xb, beta=1.0, c=G, trans=1, lower=1, overwrite_c=True)
    return G


def fit(X):
    """Train EASE and return B."""
    G = gram(X)
    G[np.diag_indices_from(G)] += LAM  # G + λI

    # P = inverse of G, done in place with Cholesky: one matrix in memory instead of three
    P, _ = lapack.spotrf(G, lower=1, overwrite_a=1)
    P, _ = lapack.spotri(P, lower=1, overwrite_c=1)
    for i in range(len(P)):  # spotri only fills the lower triangle, so copy it to the upper
        P[i, i + 1 :] = P[i + 1 :, i]

    B = P  # B = -P / diag(P) with a zero diagonal, done in place
    B /= -np.diag(B)
    np.fill_diagonal(B, 0.0)
    return B


def recommend(B, x, exclude, k=20):
    """Top k films for one user, skipping exclude. B can be read from disk."""
    scores = x.data @ B[x.indices] # == x @ B
    scores[exclude] = -np.inf
    return np.argsort(-scores)[:k]


def load_data():
    """Load the films and ratings from build.py, leaving out rarely rated films and bot accounts."""
    films = pd.read_parquet(DATA_DIR / "films.parquet")
    films = films[films["n_rated"] >= MIN_RATINGS].reset_index(drop=True)
    films["idx"] = np.arange(len(films))

    # read as categories: 90M strings would need about 10 GB
    logs = pq.read_table(DATA_DIR / "ratings.parquet", read_dictionary=["user", "slug"]).to_pandas()

    n_rated = logs.dropna(subset=["rating"]).groupby("user", observed=True).size()  # rated films per user
    real_users = n_rated.index[n_rated <= MAX_FILMS]
    logs = logs[logs["user"].isin(real_users)]

    film_idx = dict(zip(films["slug"], films["idx"]))  # slug -> the film's column in the matrix
    logs = logs[logs["slug"].isin(film_idx)]
    logs["idx"] = logs["slug"].map(film_idx).astype(int)
    logs["userId"] = logs["user"].cat.codes  # each user's row in the matrix

    return films, logs[["userId", "idx", "rating", "liked"]]
