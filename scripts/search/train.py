# Train the search model: "pairs" on the laptop, then "train" on a rented GPU.

import random
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from filmmatch.paths import DATA_DIR, DEV_FROM, PAIRS, REQUEST_DIR

MIN_SIM = 0.3  # how alike two requests' answers must be to count as a pair

BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
PREFIX = "Instruct: Given a movie request, find other requests asking for the same kind of film\nQuery: "
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
WARMUP_STEPS = 100
SCALE = 20.0  # sharpens the scores before the loss
MAX_TOKENS = 128  # fits a 24 GB GPU


def load_requests():
    return pd.read_parquet(DATA_DIR / "reddit.parquet", columns=["id", "created", "title", "body", "answers", "upvotes"])


def answer_vectors(df):
    """One row per request and one column per film, holding log(1 + upvotes).
    Films everyone recommends count less. Rows have length 1, so a dot product
    is cosine similarity."""
    cols, films = pd.factorize(np.concatenate(df["answers"].to_numpy()))  # each film's column number
    rows = np.repeat(np.arange(len(df)), df["answers"].str.len())  # each answer's request
    upvotes = np.log1p(np.concatenate(df["upvotes"].to_numpy())).astype(np.float32)
    X = sp.csr_matrix((upvotes, (rows, cols)), shape=(len(df), len(films)))

    requests_per_film = np.asarray((X > 0).sum(axis=0)).ravel()
    X = X @ sp.diags(np.log(len(df) / requests_per_film).astype(np.float32))

    length = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)))
    return sp.csr_matrix(X.multiply(1 / length))


def most_similar(X, chunk=1000):
    """For each row, the most similar other row and how similar it is."""
    best = np.empty(X.shape[0], int)
    best_sim = np.empty(X.shape[0], np.float32)
    for start in range(0, X.shape[0], chunk):
        sim = (X[start : start + chunk] @ X.T).toarray()
        rows = np.arange(len(sim))
        sim[rows, start + rows] = -1  # don't match a request with itself
        best[start : start + len(sim)] = sim.argmax(axis=1)
        best_sim[start : start + len(sim)] = sim.max(axis=1)
    return best, best_sim


def pairs():
    df = load_requests()
    df = df[(df["created"] < DEV_FROM) & (df["answers"].str.len() >= 2)].reset_index(drop=True)  # older than the dev and test sets
    best, best_sim = most_similar(answer_vectors(df))

    text = (df["title"] + "\n" + df["body"]).to_numpy()
    keep = best_sim >= MIN_SIM
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"anchor": text[keep], "positive": text[best[keep]]}).to_parquet(PAIRS, index=False)
    print(f"{keep.sum()} pairs saved to {PAIRS}")


def train():
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_TOKENS
    model[0].auto_model.gradient_checkpointing_enable()  # saves GPU memory
    model.train()

    def embed(texts):
        features = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in model.preprocess(texts).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.normalize(model(features)["sentence_embedding"].float(), dim=-1)

    pairs = pd.read_parquet(PAIRS)
    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)
    batches = [order[i : i + BATCH_SIZE] for i in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE)]

    def lr_factor(step):  # warm up, then fall to 0 by the end
        return min(1.0, (step + 1) / WARMUP_STEPS) * max(0.0, 1 - step / len(batches))

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    for step, batch in enumerate(batches, 1):
        rows = pairs.iloc[batch]
        queries = embed([PREFIX + t for t in rows["anchor"]])
        partners = embed(rows["positive"].tolist())

        # each query should score its own partner higher than the other partners in the batch
        scores = queries @ partners.T * SCALE
        labels = torch.arange(len(rows), device="cuda")
        loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2

        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(f"step {step}/{len(batches)} loss {loss.item():.3f}", flush=True)

    # embed every request and save the index
    model.eval()
    reqs = load_requests()
    texts = (reqs["title"] + "\n" + reqs["body"]).tolist()  # only new queries get the prefix
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REQUEST_DIR / "embeddings.npy", np.asarray(emb, np.float32))
    reqs.drop(columns=["title", "body"]).to_parquet(REQUEST_DIR / "requests.parquet", index=False)
    model.prompts["query"] = PREFIX  # search.py puts this in front of every query (prompt_name="query")
    model.save(str(REQUEST_DIR / "model"))
    print(f"saved to {REQUEST_DIR}")


def main():
    if sys.argv[1] == "pairs":
        pairs()
    else:
        train()


if __name__ == "__main__":
    main()
