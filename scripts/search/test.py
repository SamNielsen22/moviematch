# Test the search on real r/MovieSuggestions requests: do the films people recommended come back?
# Add "test" to run on the newest requests (once, at the end).

import sys

import numpy as np
import pandas as pd

from filmmatch.paths import DATA_DIR, DEV_FROM, TEST_FROM
from filmmatch.search import FILMS, search, use_requests_before

K = 20  # results per request
MIN_ANSWERS = 3  # a request needs this many answers the search can return
RARE_TOP = 100


def load_requests(start, end):
    """The requests posted between start and end, and every film ranked by how
    often the subreddit recommended it before start."""
    known = set(FILMS.index)
    every = pd.read_parquet(DATA_DIR / "reddit.parquet")
    every["answers"] = every["answers"].map(lambda a: [s for s in a if s in known])  # only films the search can return

    tested = every[every["created"] >= start]
    if end:
        tested = tested[tested["created"] < end]
    tested = tested[tested["answers"].str.len() >= MIN_ANSWERS]
    favorites = every[every["created"] < start]["answers"].explode().value_counts().index
    return tested, favorites


def score(found, answers, rare):
    """(hit, recall, rare recall) for one list of found films."""
    hits = found & answers
    rare_recall = len(found & rare) / min(len(rare), K) if rare else np.nan
    return bool(hits), len(hits) / min(len(answers), K), rare_recall


def interval(x, n=1000):
    """Mean with a 95% bootstrap interval."""
    x = np.asarray(x, float)
    means = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return f"{x.mean():.3f} [{np.percentile(means, 2.5):.3f}, {np.percentile(means, 97.5):.3f}]"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if split == "dev":
        start, end = DEV_FROM, TEST_FROM
    else:
        start, end = TEST_FROM, None
    requests, favorites = load_requests(start, end)
    use_requests_before(start)

    baseline = set(favorites[:K])
    common = set(favorites[:RARE_TOP])
    print(f"{len(requests)} {split} requests")

    scores = {"search": [], "baseline": []}
    for i, r in enumerate(requests.itertuples(), 1):
        answers = set(r.answers)
        rare = answers - common
        found = search(f"{r.title}\n{r.body}", K, exclude=r.mentioned)
        scores["search"].append(score(set(found["slug"]), answers, rare))
        scores["baseline"].append(score(baseline, answers, rare))
        if i % 100 == 0:
            print(f"{i}/{len(requests)}", flush=True)

    for name, rows in scores.items():
        hit, recall, rare = np.array(rows).T
        print(name, "hit", interval(hit), "recall", interval(recall), "rare recall", interval(rare[~np.isnan(rare)]))


if __name__ == "__main__":
    main()
