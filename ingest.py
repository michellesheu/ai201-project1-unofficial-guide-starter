"""
ingest.py — Milestone 3: Document Ingestion + Chunking

Loads JSONL/JSON review records from documents/, cleans them, and writes
chunks to chunks.jsonl for Milestone 4 (embedding) to consume.

Input format (each record):
    {"text": "...", "professor": "...", "dept": "...", "source": "...", "rating": 4.5}
    Required: text. All other fields optional.

Output (chunks.jsonl):
    {"id": "ethan_miller_0", "text": "Ethan Miller — CS\nGreat lecturer...",
     "review_text": "Great lecturer...", "professor": "Ethan Miller",
     "dept": "CS", "source": "https://...", "rating": 4.5,
     "professor_review_count": 12}

Chunking strategy (matches planning.md spec):
    - Per-review chunks are ATOMIC (no overlap). One review → one chunk.
    - FALLBACK only: reviews longer than MAX_REVIEW_CHARS are split into
      overlapping windows (FALLBACK_OVERLAP_CHARS). Each window is still
      prefixed so it stands alone.
"""

import argparse
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Tunable constants — adjust to match your corpus if needed
# ---------------------------------------------------------------------------

# Reviews longer than this get windowed (≈512 tokens @ 4 chars/token).
# Short RMP reviews (~50-150 words) will almost never hit this.
MAX_REVIEW_CHARS = 2000

# Window size for long reviews (fallback path only).
FALLBACK_WINDOW_CHARS = 2000

# Overlap carried from one window into the next (fallback path only).
# Per-review chunks (the normal path) have NO overlap — they are atomic.
FALLBACK_OVERLAP_CHARS = 50

# ---------------------------------------------------------------------------
# Nav/boilerplate line patterns to strip before chunking
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*rate my professors", re.IGNORECASE),
    re.compile(r"^\s*sign (in|up)", re.IGNORECASE),
    re.compile(r"^\s*log (in|out)", re.IGNORECASE),
    re.compile(r"^\s*back to (top|search)", re.IGNORECASE),
    re.compile(r"^\s*helpful\?", re.IGNORECASE),
    re.compile(r"^\s*(yes|no)\s*\(\d+\)", re.IGNORECASE),
    re.compile(r"^\s*\d+ (students|people) found this helpful", re.IGNORECASE),
    re.compile(r"^\s*flag (this)? review", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(docs_dir: str | Path) -> list[dict]:
    """
    Load review records from all *.jsonl and *.json files in docs_dir.

    JSONL: one JSON object per line (blank lines and parse errors skipped).
    JSON: top-level array of objects.

    Each record gets a '_source_file' key (stem of the file, for id slugs).
    Returns a flat list of dicts.
    """
    docs_path = Path(docs_dir)
    records: list[dict] = []

    for jsonl_file in sorted(docs_path.glob("*.jsonl")):
        loaded = 0
        with jsonl_file.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        print(f"  [warn] {jsonl_file.name}:{lineno} — expected object, got {type(obj).__name__}, skipping", file=sys.stderr)
                        continue
                    obj["_source_file"] = jsonl_file.stem
                    records.append(obj)
                    loaded += 1
                except json.JSONDecodeError as exc:
                    print(f"  [warn] {jsonl_file.name}:{lineno} — JSON parse error: {exc}, skipping", file=sys.stderr)
        print(f"  Loaded {loaded} records from {jsonl_file.name}")

    for json_file in sorted(docs_path.glob("*.json")):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"  [warn] {json_file.name} — expected top-level array, got {type(data).__name__}, skipping", file=sys.stderr)
                continue
            loaded = 0
            for i, obj in enumerate(data):
                if not isinstance(obj, dict):
                    print(f"  [warn] {json_file.name}[{i}] — expected object, skipping", file=sys.stderr)
                    continue
                obj["_source_file"] = json_file.stem
                records.append(obj)
                loaded += 1
            print(f"  Loaded {loaded} records from {json_file.name}")
        except json.JSONDecodeError as exc:
            print(f"  [warn] {json_file.name} — JSON parse error: {exc}, skipping", file=sys.stderr)

    return records


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(s: str) -> str:
    """
    Clean a review string:
      1. Unescape HTML entities (&amp; → &, etc.)
      2. Strip HTML tags
      3. Drop boilerplate lines (nav chrome, helpful buttons, etc.)
      4. Collapse runs of whitespace / blank lines
    Returns "" if nothing meaningful remains.
    """
    if not s or not s.strip():
        return ""

    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)  # strip HTML tags

    # Drop boilerplate lines (line-start match) AND inline fragments
    # (e.g. "...matters a lot. Sign In" where Sign In is end-of-line after HTML strip)
    lines = s.splitlines()
    cleaned_lines = [
        line for line in lines
        if not any(pat.match(line) for pat in _BOILERPLATE_PATTERNS)
    ]
    s = "\n".join(cleaned_lines)

    # Strip trailing inline nav tokens that survive HTML removal
    _INLINE_NAV = re.compile(
        r"\s*(sign in|sign up|log in|log out|rate my professors|"
        r"helpful\?.*|yes \(\d+\)|no \(\d+\)|flag (this )?review)\s*$",
        re.IGNORECASE,
    )
    s = _INLINE_NAV.sub("", s)

    # Collapse whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s


def _dedup_key(s: str) -> str:
    """Stable key for exact-duplicate detection (lowercase, whitespace-normalized)."""
    normalized = re.sub(r"\s+", " ", s.lower()).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fallback splitter (long reviews only)
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_long_text(text: str) -> list[str]:
    """
    Split a single long review into overlapping windows.
    Used ONLY when len(text) > MAX_REVIEW_CHARS.

    Strategy: accumulate sentences up to FALLBACK_WINDOW_CHARS, then
    carry FALLBACK_OVERLAP_CHARS from the end of the window into the next.
    Any single sentence that exceeds FALLBACK_WINDOW_CHARS is hard-split.
    """
    sentences = _SENTENCE_END.split(text)
    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    overlap_carry = ""

    for sent in sentences:
        # Hard-split an oversized sentence
        if len(sent) > FALLBACK_WINDOW_CHARS:
            # flush current first
            if current:
                windows.append(overlap_carry + " ".join(current))
                overlap_carry = (" ".join(current))[-FALLBACK_OVERLAP_CHARS:]
                current, current_len = [], 0
            for i in range(0, len(sent), FALLBACK_WINDOW_CHARS):
                chunk = sent[i : i + FALLBACK_WINDOW_CHARS]
                windows.append(overlap_carry + chunk)
                overlap_carry = chunk[-FALLBACK_OVERLAP_CHARS:]
            continue

        if current_len + len(sent) + 1 > FALLBACK_WINDOW_CHARS and current:
            windows.append(overlap_carry + " ".join(current))
            overlap_carry = (" ".join(current))[-FALLBACK_OVERLAP_CHARS:]
            current, current_len = [], 0

        current.append(sent)
        current_len += len(sent) + 1

    if current:
        windows.append(overlap_carry + " ".join(current))

    return [w.strip() for w in windows if w.strip()]


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    """Convert a string to a safe lowercase id prefix."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "review"


# ---------------------------------------------------------------------------
# Context prefix
# ---------------------------------------------------------------------------

def _make_prefix(record: dict) -> str:
    """
    Build the self-containment prefix prepended to every chunk's text.
    Without this, a bare review like "he grades hard" has no named subject.

    Examples:
        "Ethan Miller — Computer Science"
        "Ethan Miller"
        ""  (falls back to raw review text with no prefix)
    """
    professor = (record.get("professor") or "").strip()
    dept = (record.get("dept") or "").strip()
    if professor and dept:
        return f"{professor} — {dept}"
    if professor:
        return professor
    return ""


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def chunk_records(records: list[dict]) -> list[dict]:
    """
    Clean, deduplicate, and chunk records according to planning.md spec.

    Returns a list of chunk dicts ready for write_jsonl().
    """
    # --- First pass: compute per-professor review counts ---
    # Count non-empty, unique reviews per professor (best effort before dedup).
    prof_counts: dict[str, int] = defaultdict(int)
    for rec in records:
        text = clean_text(str(rec.get("text") or ""))
        if not text:
            continue
        prof = (rec.get("professor") or "").strip()
        if prof:
            prof_counts[prof] += 1

    # --- Second pass: chunk ---
    chunks: list[dict] = []
    seen_keys: set[str] = set()
    dropped_empty = 0
    dropped_dup = 0

    # Per-source counters for stable ids
    id_counters: dict[str, int] = defaultdict(int)

    for rec in records:
        raw_text = str(rec.get("text") or "")
        text = clean_text(raw_text)

        if not text:
            dropped_empty += 1
            continue

        key = _dedup_key(text)
        if key in seen_keys:
            dropped_dup += 1
            continue
        seen_keys.add(key)

        prefix = _make_prefix(rec)

        # Decide: atomic (normal path) vs windowed (fallback)
        if len(text) <= MAX_REVIEW_CHARS:
            windows = [text]
        else:
            windows = split_long_text(text)

        # Build id slug from professor name (preferred) or source file
        professor = (rec.get("professor") or "").strip()
        id_base = _slugify(professor) if professor else _slugify(rec.get("_source_file", "review"))

        for window in windows:
            chunk_text = f"{prefix}\n{window}" if prefix else window

            chunk: dict = {
                "id": f"{id_base}_{id_counters[id_base]}",
                "text": chunk_text,
                "review_text": window,  # raw (cleaned) review, without prefix
            }
            id_counters[id_base] += 1

            # Attach only metadata fields that are actually present
            for field in ("professor", "dept", "source"):
                val = (rec.get(field) or "").strip()
                if val:
                    chunk[field] = val

            rating = rec.get("rating")
            if rating is not None:
                try:
                    chunk["rating"] = float(rating)
                except (TypeError, ValueError):
                    pass

            if professor and professor in prof_counts:
                chunk["professor_review_count"] = prof_counts[professor]

            chunks.append(chunk)

    print(f"\n  Dropped (empty after clean): {dropped_empty}")
    print(f"  Dropped (exact duplicate):   {dropped_dup}")

    return chunks


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(chunks: list[dict], path: str | Path) -> None:
    """Write one JSON object per line to path."""
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def print_sample(chunks: list[dict], n: int = 5) -> None:
    """
    Print n representative chunks for manual self-containment inspection.

    For each, ask: does this make sense on its own?
    Can someone answer a question from this chunk alone?
    """
    print(f"\n{'='*60}")
    print(f"CHUNK SAMPLE ({min(n, len(chunks))} of {len(chunks)})")
    print(f"{'='*60}")
    sample = chunks[: n]
    for i, chunk in enumerate(sample, 1):
        print(f"\n--- Chunk {i} (id={chunk['id']}) ---")
        print(f"text:\n  {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}")
        meta = {k: v for k, v in chunk.items() if k not in ("id", "text", "review_text")}
        if meta:
            print(f"metadata: {meta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest JSONL/JSON review docs and write chunks.jsonl"
    )
    parser.add_argument(
        "--docs-dir", default="documents",
        help="Directory containing *.jsonl / *.json review files (default: documents)"
    )
    parser.add_argument(
        "--out", default="chunks.jsonl",
        help="Output path for chunks.jsonl (default: chunks.jsonl)"
    )
    parser.add_argument(
        "--sample", type=int, default=5,
        help="Number of sample chunks to print for inspection (default: 5)"
    )
    args = parser.parse_args()

    print(f"Loading records from {args.docs_dir}/...")
    records = load_records(args.docs_dir)
    print(f"  Total records loaded: {len(records)}")

    if not records:
        print("\nNo records found. Add *.jsonl or *.json files to documents/ and re-run.")
        sys.exit(0)

    print("\nChunking...")
    chunks = chunk_records(records)
    print(f"  Total chunks produced: {len(chunks)}")

    write_jsonl(chunks, args.out)
    print(f"\nWrote {len(chunks)} chunks → {args.out}")

    if args.sample > 0 and chunks:
        print_sample(chunks, n=args.sample)


if __name__ == "__main__":
    main()
