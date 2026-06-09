"""
scrape.py — Fetch reviews from Rate My Professors and r/UCSC.

Writes one JSONL file per source into documents/:
    documents/rmp_<slug>.jsonl   — one review per line
    documents/reddit_ucsc.jsonl  — one post/comment per line

Usage:
    python3 scrape.py

Requires: requests  (pip install requests)
"""

import base64
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed. Run: pip install requests")

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Rate My Professors — unofficial GraphQL API
# The base64 auth token is public / well-known in RMP scraping projects.
# ---------------------------------------------------------------------------

RMP_GRAPHQL_URL = "https://www.ratemyprofessors.com/graphql"
RMP_AUTH = "Basic " + base64.b64encode(b"test:test").decode()

RMP_RATINGS_QUERY = """
query RatingsListQuery($id: ID!, $count: Int!, $cursor: String) {
  node(id: $id) {
    ... on Teacher {
      id
      firstName
      lastName
      department
      school { name }
      ratings(first: $count, after: $cursor) {
        edges {
          node {
            comment
            qualityRating
            date
            class
            wouldTakeAgain
            difficultyRating
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

# Professors from planning.md Documents table
RMP_PROFESSORS = [
    {"name": "Ethan Miller",     "id": 136264,  "slug": "ethan_miller"},
    {"name": "A.M. Darke",       "id": 2463375, "slug": "am_darke"},
    {"name": "Ryan Coonerty",    "id": 439427,  "slug": "ryan_coonerty"},
    {"name": "Jesse Kass",       "id": 2895784, "slug": "jesse_kass"},
    {"name": "Anne Sizemore",    "id": 2989548, "slug": "anne_sizemore"},
    {"name": "Scott Anderson",   "id": 2319462, "slug": "scott_anderson"},
    {"name": "Edward Migliore",  "id": 218123,  "slug": "edward_migliore"},
    {"name": "Steven Owen",      "id": 2346100, "slug": "steven_owen"},
]


def _rmp_node_id(numeric_id: int) -> str:
    """RMP GraphQL node IDs are base64("Teacher-{id}")."""
    return base64.b64encode(f"Teacher-{numeric_id}".encode()).decode()


def fetch_rmp_professor(prof: dict, page_size: int = 100) -> list[dict]:
    """
    Fetch all ratings for one professor via RMP GraphQL.
    Pages through results automatically.
    Returns a list of record dicts ready for JSONL output.
    """
    node_id = _rmp_node_id(prof["id"])
    headers = {
        "Authorization": RMP_AUTH,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (academic research project)",
        "Referer": f"https://www.ratemyprofessors.com/professor/{prof['id']}",
        "Origin": "https://www.ratemyprofessors.com",
    }

    records: list[dict] = []
    cursor = None
    page = 0

    while True:
        variables = {"id": node_id, "count": page_size}
        if cursor:
            variables["cursor"] = cursor

        try:
            resp = requests.post(
                RMP_GRAPHQL_URL,
                headers=headers,
                json={"query": RMP_RATINGS_QUERY, "variables": variables},
                timeout=20,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [error] {prof['name']} page {page}: {exc}", file=sys.stderr)
            break

        data = resp.json()
        teacher = data.get("data", {}).get("node", {})
        if not teacher:
            print(f"  [warn] {prof['name']}: no data returned", file=sys.stderr)
            break

        dept = teacher.get("department") or ""
        school = (teacher.get("school") or {}).get("name") or "UCSC"
        ratings_block = teacher.get("ratings", {})

        for edge in ratings_block.get("edges", []):
            node = edge.get("node", {})
            comment = (node.get("comment") or "").strip()
            if not comment:
                continue
            record: dict = {
                "professor": prof["name"],
                "dept": dept,
                "source": f"https://www.ratemyprofessors.com/professor/{prof['id']}",
                "text": comment,
            }
            rating = node.get("qualityRating")
            if rating is not None:
                try:
                    record["rating"] = float(rating)
                except (TypeError, ValueError):
                    pass
            if node.get("class"):
                record["class"] = node["class"]
            records.append(record)

        page_info = ratings_block.get("pageInfo", {})
        if page_info.get("hasNextPage"):
            cursor = page_info["endCursor"]
            page += 1
            time.sleep(0.5)  # be polite
        else:
            break

    return records


# ---------------------------------------------------------------------------
# Reddit — public JSON API, no auth needed for read
# ---------------------------------------------------------------------------

REDDIT_HEADERS = {
    "User-Agent": "python:ucsc-rag-project:v1.0 (academic research)",
}

REDDIT_SEARCH_QUERIES = [
    "professor recommend",
    "professor avoid",
    "easy GE",
    "class review",
    "instructor",
]


def _clean_reddit_text(s: str) -> str:
    """Remove Reddit markdown formatting that adds noise."""
    if not s:
        return ""
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # [text](url) → text
    s = re.sub(r"https?://\S+", "", s)               # bare URLs
    s = re.sub(r"[*_~`>#]+", "", s)                  # markdown symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_reddit_ucsc(max_posts: int = 200) -> list[dict]:
    """
    Fetch top posts from r/UCSC using Reddit's public JSON API.
    Also pulls top-level comments from each post.
    Returns records suitable for JSONL output.
    """
    seen_ids: set[str] = set()
    records: list[dict] = []
    base_url = "https://www.reddit.com/r/UCSC"

    # 1. Search for professor/class-related posts
    for query in REDDIT_SEARCH_QUERIES:
        url = f"{base_url}/search.json"
        params = {"q": query, "sort": "top", "t": "all", "limit": 25, "restrict_sr": 1}
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [warn] Reddit search '{query}': {exc}", file=sys.stderr)
            continue

        posts = resp.json().get("data", {}).get("children", [])
        for child in posts:
            post = child.get("data", {})
            post_id = post.get("id", "")
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            title = _clean_reddit_text(post.get("title", ""))
            body = _clean_reddit_text(post.get("selftext", ""))
            combined = f"{title}. {body}".strip(" .") if body else title
            if not combined or combined == "[deleted]" or combined == "[removed]":
                continue

            records.append({
                "source": f"https://www.reddit.com/r/UCSC/comments/{post_id}/",
                "text": combined,
                "subreddit": "r/UCSC",
                "type": "post",
            })

        time.sleep(1)  # Reddit rate limit: ~1 req/sec for unauthenticated

        if len(seen_ids) >= max_posts:
            break

    # 2. Pull top-level comments from collected posts (first 20 posts only)
    comment_count = 0
    for rec in records[:20]:
        post_id = rec["source"].rstrip("/").split("/")[-1]
        url = f"{base_url}/comments/{post_id}.json"
        try:
            resp = requests.get(url, headers=REDDIT_HEADERS, params={"limit": 20}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [warn] Reddit comments {post_id}: {exc}", file=sys.stderr)
            continue

        try:
            listing = resp.json()
            comment_listing = listing[1]["data"]["children"] if len(listing) > 1 else []
        except (IndexError, KeyError, TypeError):
            continue

        for child in comment_listing:
            cdata = child.get("data", {})
            body = _clean_reddit_text(cdata.get("body", ""))
            if not body or body in ("[deleted]", "[removed]") or len(body) < 30:
                continue
            records.append({
                "source": rec["source"],
                "text": body,
                "subreddit": "r/UCSC",
                "type": "comment",
            })
            comment_count += 1

        time.sleep(0.5)

    return records


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- RMP professors ---
    print("Scraping Rate My Professors...")
    for prof in RMP_PROFESSORS:
        print(f"  {prof['name']}...", end=" ", flush=True)
        records = fetch_rmp_professor(prof)
        print(f"{len(records)} reviews")
        out_path = DOCS_DIR / f"rmp_{prof['slug']}.jsonl"
        write_jsonl(records, out_path)
        time.sleep(1)

    # --- Reddit ---
    print("\nScraping r/UCSC...")
    reddit_records = fetch_reddit_ucsc()
    print(f"  {len(reddit_records)} posts/comments collected")
    write_jsonl(reddit_records, DOCS_DIR / "reddit_ucsc.jsonl")

    print("\nDone. Run python3 ingest.py to chunk and prepare for embedding.")


if __name__ == "__main__":
    main()
