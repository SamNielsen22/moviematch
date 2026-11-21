# Find films for a description.

import sys

from filmmatch.search import search


def main():
    request = sys.argv[1]
    films = search(request, 20)
    for i, film in enumerate(films.itertuples(), 1):
        print(f"{i}: {film.title} ({film.year}) ★ {film.rating} {film.genres}")
        print( str(film.synopsis)[:84])
        print()


if __name__ == "__main__":
    main()
