# Search for films from a description.
# Queries are matched with the most similar past requests using a fine-tuned embedding model),
# then films recommended on those requests are returned.

from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from filmmatch.paths import CATALOG_DIR, REQUEST_DIR

NEIGHBOURS = 50  # past requests a new one borrows from

FILMS = pd.read_parquet(CATALOG_DIR / "films.parquet").set_index("slug")# films the search can return
REQUESTS = pd.read_parquet(REQUEST_DIR / "requests.parquet") # past requests and the films recommended on them
EMBEDDINGS = np.load(REQUEST_DIR / "embeddings.npy") # request embeddings
MODEL = SentenceTransformer(str(REQUEST_DIR / "model"), model_kwargs={"torch_dtype": "float32"})


def request_films(query, exclude):

    query_embedding = MODEL.encode(query, prompt_name="query")
    simScores = EMBEDDINGS @ query_embedding
    return score_films(simScores, exclude)


def score_films(simScores, exclude):
    """Score the films recommended on the past requests most similar to a request."""
    nearest = REQUESTS.assign(sim=simScores).nlargest(NEIGHBOURS, "sim")
    score = Counter()

    for request in nearest.itertuples():
        weight = max(request.sim, 0.0) ** 8 # TODO: check if this is optimal
        for slug, votes in zip(request.answers, request.upvotes):
            if slug in FILMS.index and slug not in exclude:
                score[slug] += weight * float(np.log1p(votes))

    return score


def search(query, k=20, exclude=[]):
    """The top k films for a request. exclude: films never to return."""
    scores = request_films(query, exclude)
    top = scores.most_common(k)
    best = [slug for slug, score in top]
    return FILMS.loc[best].reset_index()


def use_requests_before(date):
    """Search only past requests posted before date (tests use this so they can't see the answers)."""
    global REQUESTS, EMBEDDINGS
    past = (REQUESTS["created"] < date).to_numpy()
    REQUESTS, EMBEDDINGS = REQUESTS[past], EMBEDDINGS[past]
