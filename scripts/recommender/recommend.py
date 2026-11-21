# Recommend films for a Letterboxd user.
# An optional number after the username skips that many of the most popular films.

import sys

import numpy as np
import pandas as pd

from filmmatch.letterboxd import fetch_films
from filmmatch.paths import DATA_DIR
from filmmatch.recommender import build_matrix, recommend


def main():
    username = sys.argv[1]
    skip_top = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    films = pd.read_parquet(DATA_DIR / "model_films.parquet")
    B = np.load(DATA_DIR / "model.npy", mmap_mode="r")  # read from disk as needed
    tv = np.flatnonzero(films["is_tv"])
    popularity = films["n_rated"].to_numpy(np.float32)

    profile = pd.DataFrame(fetch_films(username))
    seen = profile.merge(films[["slug", "idx"]], on="slug")  # the profile's films the model knows
    print(f"{len(seen)} of the {len(profile)} films on the profile are in the model")

    x = build_matrix(seen.assign(userId=0), len(films))  # one row: this user
    most_popular = np.argsort(-popularity)[:skip_top]
    exclude = np.concatenate([seen["idx"].to_numpy(), tv, most_popular])
    top = recommend(B, x, exclude)

    print()
    for i, idx in enumerate(top, 1):
        film = films.iloc[idx]
        print(f"{i}: {film.title} ({film.year})")


if __name__ == "__main__":
    main()
