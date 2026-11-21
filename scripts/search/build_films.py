# Build the list of films the search can return.

import argparse
import json

import pandas as pd

from filmmatch.paths import CATALOG_DIR, DATA_DIR, LETTERBOXD_DUMP


def to_row(film, slug):
    """One film from the dump as a row of the catalog, or None if it has no synopsis."""
    synopsis = (film.get("synopsis") or "").strip()
    if not synopsis:
        return None
    year = film.get("year")
    rating = film.get("rating") or ""  # text that starts with the average stars
    return {
        "slug": slug,
        "title": film.get("title"),
        "year": int(year) if str(year).isdigit() else None,
        "genres": ", ".join(film.get("genres") or []),
        "rating": float(rating.split()[0]) if rating[:1].isdigit() else None,
        "synopsis": synopsis,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ratings", type=int, help="only films with this many ratings in the crawl")
    args = ap.parse_args()

    crawled = pd.read_parquet(DATA_DIR / "films.parquet", columns=["slug", "n_rated", "is_tv"])
    tv = set(crawled.loc[crawled["is_tv"], "slug"])
    allowed = set(crawled.loc[crawled["n_rated"] >= args.min_ratings, "slug"]) if args.min_ratings else None

    rows = {}
    with LETTERBOXD_DUMP.open() as fh:
        for line in fh:
            film = json.loads(line)
            slug = film["url"].rstrip("/").rsplit("/", 1)[-1]
            if slug in rows or slug in tv or (allowed is not None and slug not in allowed):
                continue  # the dump lists some films twice; the first copy wins
            row = to_row(film, slug)
            if row:
                rows[slug] = row

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    films = pd.DataFrame(rows.values()).astype({"year": "Int64"})
    films.to_parquet(CATALOG_DIR / "films.parquet", index=False)
    print(f"{len(films):,} films -> {CATALOG_DIR / 'films.parquet'}")


if __name__ == "__main__":
    main()
