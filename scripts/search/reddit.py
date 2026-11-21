# Pull movie requests and the films people recommended out of r/MovieSuggestions.
# A reply only counts a film when a whole line of it is the film's title.

import json
import re
import unicodedata
from collections import Counter

import pandas as pd

from filmmatch.paths import DATA_DIR, REDDIT_DIR

OUT = DATA_DIR / "reddit.parquet"
MIN_RATED = 20       # a film needs this many ratings to count
MIN_TITLE_LEN = 3    # shorter titles ("Up", "It") need a year
REMOVED = ("[removed]", "[deleted]")
# titles that are also common replies; only matched with a year
PLAIN_REPLIES = set(p.strip() for p in """it, up, her, us, yes, no, ok, okay, this, that, thanks, thank you, same, nope,
lol, cool, nice, wow, oh, what, why, wait, and, me too, not, sorry, true, agreed, both, all, good, great, nothing,
something, anything, everything, one, two, you, him, them, they, we, i, mom, dad, home, lmao, haha, seen it, this one,
the same, hell yes, yes please, the end, fin, love, hate, again, more, please, now, then, here, there, tomorrow, today,
tonight, yesterday, life, mother, father, brother, sister, friends, family, the thing is, the movie, the film,
the one, the other one, edit, spoiler, spoilers, same here, exactly, absolutely, definitely, this guy, second this,
seconded, bump, following, saved, saving, x, xx, xxx""".split(","))

YEAR_RE = re.compile(r"^(.+?)\s*[\[(]\s*((?:19|20)\d\d)\s*[\])]")  # "Heat (1995)"
LIST_MARK_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•>+#]+)\s*")  # "1." or "-" at the start of a line
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url)
EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|_(.+?)_")  # **Title** in running text


def norm(text):
    """Lowercase words without accents or punctuation, so "Amélie!" matches "amelie"."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def strip_markup(text):
    return text.strip().strip("*_~`\"'")


def short_titles(films, norm):
    """Lookups for films people name without the subtitle, like "Kill Bill" for
    "Kill Bill: Vol. 1". Only the most popular film starting with that name
    counts, so "Harry Potter" doesn't match a documentary."""
    first = {}  # first words of a title -> most popular film starting with them
    for t, slug in zip(films["title"].map(norm), films["slug"]):
        words = t.split()
        for i in range(1, len(words) + 1):
            first.setdefault(" ".join(words[:i]), slug)
    short, short_year = {}, {}
    for f in films[films["title"].str.contains(":", regex=False)].itertuples():
        s = norm(f.title.split(":")[0])
        if s and first.get(s) == f.slug:
            short.setdefault(s, f.slug)
            if not pd.isna(f.year):
                short_year.setdefault((s, int(f.year)), f.slug)
    return short, short_year


def load_catalog():
    """Lookups from title, and title + year, to film, with and without subtitles.
    When films share a title, the most popular one wins."""
    films = pd.read_parquet(DATA_DIR / "films.parquet")
    films = films[(films["n_rated"] >= MIN_RATED) & ~films["is_tv"]].sort_values("n_rated", ascending=False)
    by_title, by_title_year = {}, {}
    for film in films.itertuples():
        title = norm(film.title)
        if len(title) >= MIN_TITLE_LEN:
            by_title.setdefault(title, film.slug)
        if not pd.isna(film.year):
            by_title_year.setdefault((title, int(film.year)), film.slug)
    short, short_year = short_titles(films, norm)
    short = {s: slug for s, slug in short.items() if len(s) >= MIN_TITLE_LEN}
    print(f"catalog: {len(by_title)} titles, {len(short)} without subtitle")
    return by_title, by_title_year, short, short_year


def match_title(text, catalog):
    """The film this text is the title of, or None."""
    by_title, by_title_year, short, short_year = catalog
    m = YEAR_RE.match(text)
    if m:
        title, year = norm(m.group(1)), int(m.group(2))
        for lookup in (by_title_year, short_year):
            for y in (year, year - 1, year + 1):  # sites sometimes disagree on the year by one
                if (title, y) in lookup:
                    return lookup[(title, y)]
        return None

    title = norm(text)
    if title in PLAIN_REPLIES:
        return None
    for lookup in (by_title, short):  # full titles first
        for t in (title, "the " + title):  # "Goonies" for "The Goonies"
            if t in lookup:
                return lookup[t]
    return None


def films_in(text, catalog):
    """Every film a post or reply names."""
    found = set()
    for m in EMPHASIS_RE.finditer(text):
        slug = match_title(next(g for g in m.groups() if g), catalog)
        if slug:
            found.add(slug)

    for line in text.splitlines():
        line = strip_markup(LINK_RE.sub(r"\1", LIST_MARK_RE.sub("", line)))
        if not line:
            continue
        slug = match_title(line, catalog)
        if slug:
            found.add(slug)
        elif "," in line or ";" in line:  # "Con Air, The Rock, Face/Off"
            for piece in re.split(r"[,;]", line):
                slug = match_title(strip_markup(piece), catalog)
                if slug:
                    found.add(slug)
    return found


def read_jsonl(path):
    """Each line of the file as a dict, skipping broken lines (like a half-downloaded last one)."""
    bad = 0
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  {path.name}: skipped {bad} unparsable line(s)")


def is_request(post):
    """A text post asking for suggestions: flaired as a request, or older and unflaired."""
    body = (post.get("selftext") or "").strip()
    flair = (post.get("link_flair_text") or "").upper()
    if not post.get("is_self") or body in REMOVED or post.get("author") == "AutoModerator":
        return False
    return not flair or "REQUEST" in flair


def read_requests(catalog):
    """Every request post by id, with the films it names itself and an empty answers counter."""
    posts = {}
    n_read = 0
    for post in read_jsonl(REDDIT_DIR / "r_MovieSuggestions_posts.jsonl"):
        n_read += 1
        if not is_request(post):
            continue
        body = (post.get("selftext") or "").strip()
        flair = post.get("link_flair_text") or ""
        posts[post["id"]] = {
            "id": post["id"],
            "created": int(post["created_utc"]),
            "score": int(post.get("score") or 0),
            "title": post["title"].strip(),
            "body": body,
            "flaired": bool(flair),
            "mentioned": films_in(post["title"] + "\n" + body, catalog),
            "answers": Counter(),
        }
    print(f"{n_read} posts read, {len(posts)} requests kept")
    return posts


def add_answers(requests, catalog):
    """Count the films each direct reply names toward its request, weighted by the
    reply's upvotes. Films the request names itself don't count. Returns how many
    replies named each film."""
    seen = set()
    n_read = n_replies = n_naming = 0
    times_named = Counter()
    for path in sorted(REDDIT_DIR.glob("r_MovieSuggestions_comments*.jsonl")):
        for reply in read_jsonl(path):
            n_read += 1
            if reply["id"] in seen or not reply["parent_id"].startswith("t3_"):
                continue  # already counted, or a reply to another comment
            seen.add(reply["id"])
            request = requests.get(reply["link_id"][3:])
            body = reply.get("body") or ""
            if request is None or body in REMOVED or reply.get("author") == "AutoModerator" or reply.get("is_submitter"):
                continue
            n_replies += 1
            films = films_in(body, catalog) - request["mentioned"]
            if films:
                n_naming += 1
                upvotes = max(int(reply.get("score") or 1), 1)
                for slug in films:
                    request["answers"][slug] += upvotes
                    times_named[slug] += 1
        print(f"{path.name}: {n_read} comments so far")
    print(f"{n_replies} replies to requests, {n_naming} named a film")
    return times_named


def to_table(requests):
    """Requests that got at least one answer, oldest first, answers best first."""
    rows = []
    for r in requests.values():
        if not r["answers"]:
            continue
        ranked = sorted(r["answers"].items(), key=lambda kv: -kv[1])
        rows.append({
            "id": r["id"], "created": r["created"], "score": r["score"], "title": r["title"],
            "body": r["body"], "flaired": r["flaired"], "mentioned": sorted(r["mentioned"]),
            "answers": [slug for slug, _ in ranked], "upvotes": [votes for _, votes in ranked],
        })
    df = pd.DataFrame(rows).sort_values("created").reset_index(drop=True)
    df["created"] = pd.to_datetime(df["created"], unit="s")
    return df


def main():
    catalog = load_catalog()
    requests = read_requests(catalog)
    times_named = add_answers(requests, catalog)
    df = to_table(requests)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"{len(df)} requests with answers")
    print("most-named films:", times_named.most_common(15))
    for r in df.sample(3, random_state=0).itertuples():
        print(f"\n[{r.created:%Y-%m-%d}] {r.title}\n  {r.body[:200]!r}\n  -> {r.answers[:8]}")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
