
from pathlib import Path

DATA_DIR = Path("data/processed")# tables and models built from the raw data
PROFILE_DIR = Path("data/cache/letterboxd")# one file per downloaded Letterboxd user
FILM_PAGE_DIR = Path("data/cache/films")# one file per downloaded film page
USERS_FILE = Path("data/cache/usernames.txt")# usernames to download
REDDIT_DIR = Path("data/raw/reddit")# r/MovieSuggestions dumps
LETTERBOXD_DUMP = Path("data/raw/letterboxd_meta/full_dump.jsonl")# synopses and genres
CATALOG_DIR = DATA_DIR / "text"# films the search can return
REQUEST_DIR = DATA_DIR / "requests_qwen06_ft"# past requests and the search model
PAIRS = Path("data/cache/embedder_pairs.parquet")# training pairs for the search model

# The newest Reddit requests are kept for testing the search, about 500 in each set
DEV_FROM = "2025-04-06"# dev set: for trying changes
TEST_FROM = "2025-04-15"# sealed test set: run once at the end
