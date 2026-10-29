# Download from Letterboxd, saved so nothing is downloaded twice.

import json

import pandas as pd
from letterboxdpy.members import Members
from letterboxdpy.pages.movie_profile import MovieProfile
from letterboxdpy.pages.user_films import UserFilms

from filmmatch.paths import FILM_PAGE_DIR, PROFILE_DIR

MEMBERS_PER_PAGE = 25


def fetch_films(username):
    """A user's films: slug, title, year, rating (stars or None) and liked."""
    cache = PROFILE_DIR / f"{username}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    movies = UserFilms(username).get_films()["movies"]
    films = [
        {
            "slug": movie["slug"],
            "title": movie["name"],
            "year": movie["year"],
            "rating": movie["rating"],
            "liked": movie.get("liked", False),
        }
        for movie in movies.values()
    ]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(films, indent=1))
    return films


def fetch_film(slug):
    """A film's title, year and whether it's a movie or TV show."""
    cache = FILM_PAGE_DIR / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    page = MovieProfile(slug)
    tmdb_link = page.get_tmdb_link() or ""
    film = {
        "slug": slug,
        "title": page.get_title(),
        "year": page.get_year(),
        "tmdb_type": "tv" if "/tv/" in tmdb_link else "movie",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(film, indent=1))
    return film


def load_films():
    """Every cached film page as one table."""
    rows = [json.loads(f.read_text()) for f in FILM_PAGE_DIR.glob("*.json")]
    return pd.DataFrame(rows)


def film_members(slug, pages=2):
    """Usernames from a film's members pages (25 per page)."""
    return list(Members(f"https://letterboxd.com/film/{slug}/members/", max=MEMBERS_PER_PAGE * pages).get_members())
