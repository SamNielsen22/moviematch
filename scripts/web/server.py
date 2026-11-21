# The API behind the website in web/.
# Run from the project folder: uvicorn server:app --app-dir scripts/web

import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from letterboxdpy.pages.movie_profile import MovieProfile

from filmmatch.letterboxd import fetch_films
from filmmatch.paths import DATA_DIR, POSTER_DIR
from filmmatch.recommender import build_matrix, recommend
from filmmatch.search import FILMS as CATALOG  # films the search can return, with synopses and posters
from filmmatch.search import search

MOOD_POOL = 100  # the mood's top films that taste reorders
TASTE_WEIGHT = 0.5  # 0 = mood only, 1 = taste only (not tuned yet)
COLUMNS = ["slug", "title", "year", "genres", "rating", "synopsis", "poster_url"]

app = FastAPI()

FILMS = pd.read_parquet(DATA_DIR / "model_films.parquet")  # films the recommender knows
B = np.load(DATA_DIR / "model.npy", mmap_mode="r")  # read from disk as needed
TV = np.flatnonzero(FILMS["is_tv"])
FILM_IDX = FILMS.set_index("slug")["idx"]  # slug -> the film's row in B


def newer_poster(slug):
    """The poster link new_poster() downloaded for a film, or None if it never had to."""
    cache = POSTER_DIR / f"{slug}.txt"
    return cache.read_text() if cache.exists() else None


def film_info(slugs):
    """What the website shows for each film, as a list of dicts with missing values as null."""
    films = CATALOG.reindex(slugs)
    titles = FILMS.set_index("slug").reindex(slugs)  # films only the recommender knows still have a title and year
    films["title"] = films["title"].fillna(titles["title"])
    films["year"] = films["year"].fillna(titles["year"])
    films = films.rename_axis("slug").reset_index()
    films["poster_url"] = [newer_poster(slug) or saved for slug, saved in zip(films["slug"], films["poster_url"])]
    return json.loads(films[COLUMNS].to_json(orient="records"))


def load_profile(username):
    """The user's Letterboxd films, or an error the website can show."""
    try:
        profile = pd.DataFrame(fetch_films(username))
    except Exception:
        raise HTTPException(404, f"Couldn't find {username} on Letterboxd")
    if profile.empty:
        raise HTTPException(404, f"{username} hasn't logged any films")
    return profile


@app.get("/api/profile")
def profile_info(username: str):
    return {"films": len(load_profile(username))}


@app.get("/api/poster")
def new_poster(slug: str):
    """A film's current poster link from its Letterboxd page. The website asks when a saved link has died."""
    if slug not in CATALOG.index:
        raise HTTPException(404, "No such film")
    if newer_poster(slug) is None:
        POSTER_DIR.mkdir(parents=True, exist_ok=True)
        (POSTER_DIR / f"{slug}.txt").write_text(MovieProfile(slug).get_poster() or "")
    return {"poster_url": newer_poster(slug)}


@app.get("/api/match")
def match(mood: str = "", username: str = ""):
    """Films for a mood, a Letterboxd user's taste, or both."""
    if not username:
        return film_info(search(mood, 20)["slug"])

    profile = load_profile(username)
    seen = profile.merge(FILMS[["slug", "idx"]], on="slug")  # the profile's films the model knows
    x = build_matrix(seen.assign(userId=0), len(FILMS))  # one row: this user
    if not mood.strip():
        exclude = np.concatenate([seen["idx"].to_numpy(), TV])
        return film_info(FILMS["slug"].iloc[recommend(B, x, exclude)])

    # both: take the mood's top films, then move the ones the user should like up
    films = search(mood, MOOD_POOL, exclude=set(profile["slug"]))
    taste = x.data @ B[x.indices]  # the scores recommend() sorts by
    known = films["slug"].isin(FILM_IDX.index)
    films["taste"] = np.nan
    films.loc[known, "taste"] = taste[FILM_IDX[films["slug"][known]].to_numpy()]

    mood_rank = 1 - np.arange(len(films)) / len(films)  # 1 for the best match
    taste_rank = films["taste"].rank(pct=True).fillna(0)  # films the model doesn't know count as least liked
    films["score"] = (1 - TASTE_WEIGHT) * mood_rank + TASTE_WEIGHT * taste_rank
    return film_info(films.nlargest(20, "score")["slug"])
