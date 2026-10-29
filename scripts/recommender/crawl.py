# Find Letterboxd usernames (discover) and download their films (crawl).
# Safe to stop and restart.

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from letterboxdpy.core.exceptions import PrivateRouteError, ResourceNotFoundError

from filmmatch.letterboxd import fetch_films, film_members
from filmmatch.paths import DATA_DIR, PROFILE_DIR, USERS_FILE

FIRST_PAUSE, LONGEST_PAUSE = 60, 3600  # seconds to wait when Letterboxd blocks us


def save_usernames(found):
    """Add usernames to USERS_FILE and return how many it holds. Re-reads the
    file first in case another run added names."""
    known = set(USERS_FILE.read_text().split()) if USERS_FILE.exists() else set()
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text("\n".join(sorted(known | found)))
    return len(known | found)


def discover(args):
    """Find usernames on the members pages of random films, taking popular and
    obscure films equally so casual users get in too."""
    films = pd.read_parquet(DATA_DIR / "films.parquet")
    films["quarter"] = (films["n_rated"].rank(pct=True) * 4).astype(int).clip(upper=3)  # 0 = least rated
    sample = films.groupby("quarter").sample(args.films // 4, random_state=args.seed)

    found = set()
    for i, slug in enumerate(sample["slug"], 1):
        try:
            found |= set(film_members(slug, args.pages))
        except Exception as e:
            print(f"{slug}: {type(e).__name__}: {e}", flush=True)
        print(f"{i}/{len(sample)} {slug}: {len(found)} users so far", flush=True)
        if i % 50 == 0:
            save_usernames(found)
    print(f"{len(found)} found, {save_usernames(found)} total in {USERS_FILE}")


def crawl(args):
    """Download profiles with several workers. If Letterboxd starts blocking us,
    every worker waits (1 minute, doubling up to 1 hour)."""
    names = USERS_FILE.read_text().split()[: args.n]
    todo = [name for name in names if not (PROFILE_DIR / f"{name}.json").exists()]
    lock = threading.Lock()
    state = {"pause": FIRST_PAUSE, "paused_until": 0.0, "done": 0}

    def log(name, message):
        with lock:
            state["done"] += 1
            print(f"{state['done']}/{len(todo)} {name}: {message}", flush=True)

    def pause_everyone(name, error):
        with lock:
            if time.time() < state["paused_until"]:
                return  # another worker already paused everyone
            wait = state["pause"]
            state["paused_until"] = time.time() + wait
            state["pause"] = min(wait * 2, LONGEST_PAUSE)
            print(f"{name}: {type(error).__name__}: {error} -- all workers waiting {wait // 60} min", flush=True)

    def download(name):
        while True:
            while time.time() < state["paused_until"]:
                time.sleep(5)
            try:
                films = fetch_films(name)
            except (ResourceNotFoundError, PrivateRouteError) as e:
                (PROFILE_DIR / f"{name}.json").write_text("[]")  # remember so we never retry it
                log(name, f"skipped ({type(e).__name__})")
                break
            except Exception as e:
                pause_everyone(name, e)
                continue
            state["pause"] = FIRST_PAUSE
            log(name, f"{len(films)} films")
            break

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, todo))


def main():
    ap = argparse.ArgumentParser()
    steps = ap.add_subparsers(dest="step", required=True)
    d = steps.add_parser("discover")
    d.add_argument("--films", type=int, default=200)
    d.add_argument("--pages", type=int, default=2)
    d.add_argument("--seed", type=int, default=0)
    c = steps.add_parser("crawl")
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    if args.step == "discover":
        discover(args)
    else:
        crawl(args)


if __name__ == "__main__":
    main()
