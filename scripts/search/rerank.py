# Train a reranker that re-orders the search's top candidates.
# "data" builds its training examples from past Reddit requests, "train" fine-tunes a reranker
# on them (on a rented GPU), and "test" scores a reranker on the dev set.

import sys

import numpy as np
import pandas as pd

from filmmatch import search
from filmmatch.paths import DATA_DIR, DEV_FROM, RERANK_PAIRS, RERANKER_DIR, TEST_FROM
from test import K, RARE_TOP, interval, load_requests, score  # scripts/search/test.py

N_REQUESTS = 20_000  # training requests to sample
CANDIDATES = 100  # search results the reranker re-orders
NEGATIVES = 40  # highest-ranked wrong films kept per request

BASE_RERANKER = "BAAI/bge-reranker-v2-m3"
MAX_TOKENS = 512  # request and film text together
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# listwise training: each right answer is scored against GROUP - 1 wrong films at once
GROUP = 16
RIGHT_PER_REQUEST = 3
SKIP_WRONG = 5  # the search's top "wrong" films are often right answers nobody named, so they aren't used

SEARCH_WEIGHTS = np.linspace(0, 1, 11)  # test blends the search's and the reranker's scores


def request_texts():
    requests = pd.read_parquet(DATA_DIR / "reddit.parquet", columns=["id", "title", "body"]).set_index("id")
    return requests["title"] + "\n" + requests["body"]


def film_text(slug, note=""):
    film = search.FILMS.loc[slug]
    themes = ", ".join(film.themes[:5])  # Letterboxd's vibe tags, like "Twisted dark psychological thriller"
    return f"{note}{film.title} ({film.year}). {film.genres}. {themes}. {film.synopsis}"


def candidates(simScores, exclude, titles):
    """The search's top films with their scores, and a note on which similar past requests recommended each."""
    top = search.score_films(simScores, exclude).most_common(CANDIDATES)
    nearest = search.REQUESTS.assign(sim=simScores).nlargest(search.NEIGHBOURS, "sim")
    named = {}
    for request in nearest.itertuples():
        for slug in set(request.answers):
            named.setdefault(slug, []).append(titles[request.id][:80])

    found = {}
    for slug, score in top:
        relative = round(100 * score / top[0][1])
        found[slug] = (score, f"score {relative}, named by {len(named[slug])} similar requests: {' / '.join(named[slug][:3])}. ")
    return found


def data():
    search.use_requests_before(DEV_FROM)  # only requests older than the dev and test sets
    past = search.REQUESTS.reset_index(drop=True)  # row i matches search.EMBEDDINGS[i]
    reddit = pd.read_parquet(DATA_DIR / "reddit.parquet", columns=["id", "title", "mentioned"]).set_index("id")
    sample = past.sample(N_REQUESTS, random_state=0)

    rows = []
    for n, request in enumerate(sample.itertuples(), 1):
        # the stored embedding stands in for embedding the request again, which would take hours
        simScores = search.EMBEDDINGS @ search.EMBEDDINGS[request.Index]
        simScores[request.Index] = -np.inf  # not the request itself
        found = candidates(simScores, set(reddit["mentioned"][request.id]), reddit["title"])

        answers = set(request.answers)
        right = [slug for slug in found if slug in answers]
        wrong = [slug for slug in found if slug not in answers][:NEGATIVES]
        if not right:
            continue  # nothing to learn from without a right answer
        rows += [(request.id, slug, found[slug][1], 1) for slug in right]
        rows += [(request.id, slug, found[slug][1], 0) for slug in wrong]
        if n % 1000 == 0:
            print(f"{n}/{N_REQUESTS}", flush=True)

    pairs = pd.DataFrame(rows, columns=["request", "slug", "note", "label"])
    pairs.to_parquet(RERANK_PAIRS, index=False)
    print(f"{pairs['request'].nunique()} requests, {len(pairs)} pairs ({pairs['label'].sum()} right), saved to {RERANK_PAIRS}")


def pointwise(pairs, texts, crowd):
    """Each film on its own, labelled right or wrong."""
    pairs = pairs.sample(frac=1, random_state=0)  # shuffled
    return {
        "request": [texts[p.request] for p in pairs.itertuples()],
        "film": [film_text(p.slug, p.note if crowd else "") for p in pairs.itertuples()],
        "label": [float(p.label) for p in pairs.itertuples()],
    }


def listwise(pairs, texts, crowd):
    """A right answer followed by GROUP - 1 wrong films for the same request."""
    columns = {"request": [], "right": []} | {f"wrong{i}": [] for i in range(1, GROUP)}
    for request, films in pairs.groupby("request", sort=False):
        right = films[films["label"] == 1].head(RIGHT_PER_REQUEST)
        wrong = films[films["label"] == 0].iloc[SKIP_WRONG:]
        if len(wrong) < GROUP - 1:
            continue
        for film in right.itertuples():
            picked = wrong.sample(GROUP - 1, random_state=len(columns["request"]))
            columns["request"].append(texts[request])
            columns["right"].append(film_text(film.slug, film.note if crowd else ""))
            for i, other in enumerate(picked.itertuples(), 1):
                columns[f"wrong{i}"].append(film_text(other.slug, other.note if crowd else ""))
    return columns


def train(n_requests, crowd, groups):
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss, MultipleNegativesRankingLoss

    pairs = pd.read_parquet(RERANK_PAIRS)
    used = pairs["request"].drop_duplicates().iloc[:n_requests]
    pairs = pairs[pairs["request"].isin(used)]
    model = CrossEncoder(BASE_RERANKER, num_labels=1, max_length=MAX_TOKENS)
    if groups:
        examples = Dataset.from_dict(listwise(pairs, request_texts(), crowd))
        loss = MultipleNegativesRankingLoss(model, num_negatives=0)  # pick the right film out of its group
        batch_size = 1  # one group is GROUP films, the same as a pointwise batch
    else:
        examples = Dataset.from_dict(pointwise(pairs, request_texts(), crowd))
        loss = BinaryCrossEntropyLoss(model)
        batch_size = BATCH_SIZE
    print(f"training on {len(examples)} examples from {len(used)} requests", flush=True)

    args = CrossEncoderTrainingArguments(
        output_dir=str(RERANKER_DIR / "checkpoints"), num_train_epochs=1, per_device_train_batch_size=batch_size,
        learning_rate=LEARNING_RATE, warmup_ratio=0.1, bf16=True, logging_steps=200, save_strategy="no", report_to="none")
    CrossEncoderTrainer(model=model, args=args, train_dataset=examples, loss=loss).train()
    model.save(str(RERANKER_DIR))
    print(f"saved to {RERANKER_DIR}")


def rescale(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def test(model_name, crowd):
    """Dev-set scores of the search's and the reranker's scores blended, from all reranker to all search."""
    from sentence_transformers.cross_encoder import CrossEncoder

    reranker = CrossEncoder(model_name, max_length=MAX_TOKENS)
    requests, favorites = load_requests(DEV_FROM, TEST_FROM)
    search.use_requests_before(DEV_FROM)
    titles = pd.read_parquet(DATA_DIR / "reddit.parquet", columns=["id", "title"]).set_index("id")["title"]
    common = set(favorites[:RARE_TOP])

    scores = {weight: [] for weight in SEARCH_WEIGHTS}
    for n, r in enumerate(requests.itertuples(), 1):
        text = f"{r.title}\n{r.body}"
        simScores = search.EMBEDDINGS @ search.MODEL.encode(text, prompt_name="query")
        found = candidates(simScores, set(r.mentioned), titles)
        slugs = list(found)
        searched = rescale(np.array([found[slug][0] for slug in slugs]))
        fit = rescale(reranker.predict([(text, film_text(slug, found[slug][1] if crowd else "")) for slug in slugs], batch_size=64))

        answers = set(r.answers)
        rare = answers - common
        for weight in SEARCH_WEIGHTS:
            order = np.argsort(-(weight * searched + (1 - weight) * fit), kind="stable")
            scores[weight].append(score({slugs[i] for i in order[:K]}, answers, rare))
        if n % 100 == 0:
            print(f"{n}/{len(requests)}", flush=True)

    for weight, rows in scores.items():
        hit, recall, rare = np.array(rows).T
        print(f"search weight {weight:.1f}", "hit", interval(hit), "recall", interval(recall), "rare recall", interval(rare[~np.isnan(rare)]))


def main():
    step = sys.argv[1]
    crowd = "crowd" in sys.argv  # give the reranker the search's score and the past requests that named each film
    if step == "data":
        data()
    elif step == "train":
        train(int(sys.argv[2]), crowd, "listwise" in sys.argv)
    else:
        test(sys.argv[2] if len(sys.argv) > 2 else str(RERANKER_DIR), crowd)


if __name__ == "__main__":
    main()
