# Train the recommender on every downloaded user and save it.

import numpy as np

from filmmatch.paths import DATA_DIR
from filmmatch.recommender import build_matrix, fit, load_data


def save_by_rows(path, B, block=2048):
    """Save B row by row, a block at a time to save memory. recommend.py reads
    single rows from disk, which is only fast in this layout."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=B.dtype, shape=B.shape)
    for start in range(0, len(B), block):
        out[start : start + block] = B[start : start + block]
    out.flush()
    del out


def main():
    films, logs = load_data()
    X = build_matrix(logs, len(films))
    del logs  # free memory for the fit

    B = fit(X)
    save_by_rows(DATA_DIR / "model.npy", B)
    films.to_parquet(DATA_DIR / "model_films.parquet", index=False)
    print(f"fit on {X.shape[0]} users and {len(films)} films, saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
