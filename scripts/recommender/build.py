# Combine the downloaded profiles into a ratings table and a films table.

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from filmmatch.letterboxd import fetch_film, load_films
from filmmatch.paths import DATA_DIR, FILM_PAGE_DIR, PROFILE_DIR

MIN_RATINGS = 100  # films with fewer ratings are left out
USERS_PER_PART = 2000


def to_table(rows):
    """(user, slug, rating, liked) tuples as a small table. Categories store each
    username and slug once instead of once per row."""
    part = pd.DataFrame(rows, columns=["user", "slug", "rating", "liked"])
    return part.astype({"user": "category", "slug": "category", "rating": "float32"})


def read_profiles():
    """Every downloaded profile as one ratings table, plus each film's (title, year)."""
    files = sorted(PROFILE_DIR.glob("*.json"))
    titles = {}
    parts = []  # one small table per USERS_PER_PART users, to keep memory down
    rows = []
    for i, f in enumerate(files, 1):
        for film in json.loads(f.read_text()):
            rows.append((f.stem, film["slug"], film["rating"], film["liked"]))
            titles.setdefault(film["slug"], (film["title"], film["year"]))
        if i % USERS_PER_PART == 0 or i == len(files):
            parts.append(to_table(rows))
            rows = []

    # pd.concat would turn the categories back into strings and run out of memory
    ratings = pd.DataFrame({
        "user": union_categoricals([p["user"] for p in parts]),
        "slug": union_categoricals([p["slug"] for p in parts]),
        "rating": np.concatenate([p["rating"].to_numpy() for p in parts]),
        "liked": np.concatenate([p["liked"].to_numpy() for p in parts]),
    })
    return ratings, titles, len(files)


def count_films(ratings, titles):
    """One row per film with how many users rated and logged it, most-rated first."""
    films = (
        ratings.groupby("slug", observed=True)
        .agg(n_rated=("rating", "count"), n_logged=("rating", "size"))
        .reset_index()
        .sort_values("n_rated", ascending=False, ignore_index=True)
    )
    films["slug"] = films["slug"].astype(str)
    films["title"] = films["slug"].map(lambda s: titles[s][0])
    films["year"] = films["slug"].map(lambda s: titles[s][1]).astype("Int64")
    return films[["slug", "title", "year", "n_rated", "n_logged"]]


def fetch_missing_films(slugs, workers=4):
    """Download the page of every film not cached yet."""
    todo = [s for s in slugs if not (FILM_PAGE_DIR / f"{s}.json").exists()]
    if not todo:
        return
    print(f"fetching {len(todo)} film pages", flush=True)

    def fetch_with_retries(slug):
        for attempt in range(3):
            try:
                return fetch_film(slug)
            except Exception as e:
                error = e
                time.sleep(5 * (attempt + 1))
        print(f"{slug}: {type(error).__name__}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(fetch_with_retries, todo), 1):
            if i % 1000 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


def flag_tv(films):
    """True for TV shows (TMDB lists them under /tv/)."""
    pages = load_films()
    if pages.empty:
        return pd.Series(False, index=films.index)
    tv_shows = set(pages.loc[pages["tmdb_type"] == "tv", "slug"])
    return films["slug"].isin(tv_shows)


def main():
    ratings, titles, n_users = read_profiles()
    films = count_films(ratings, titles)
    print(f"{n_users} users, {len(ratings)} logged films, {len(films)} distinct films", flush=True)

    fetch_missing_films(sorted(films.loc[films["n_rated"] >= MIN_RATINGS, "slug"]))
    films["is_tv"] = flag_tv(films)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(DATA_DIR / "ratings.parquet", index=False)
    films.to_parquet(DATA_DIR / "films.parquet", index=False)
    print(f"saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
