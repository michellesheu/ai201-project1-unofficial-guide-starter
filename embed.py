"""
embed.py — Milestone 4: Embedding + Vector Store

Loads chunks from chunks.jsonl, embeds with all-MiniLM-L6-v2, and stores
in a persistent ChromaDB collection. Safe to re-run: checks for existing
IDs and skips already-embedded chunks.

Usage:
    python3 embed.py                        # embed chunks.jsonl → chroma_db/
    python3 embed.py --chunks chunks.jsonl  # explicit paths
    python3 embed.py --reset                # wipe collection and re-embed everything

Retrieval (import from other scripts):
    from embed import retrieve
    results = retrieve("Does Ethan Miller curve exams?", top_k=5)
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME   = "all-MiniLM-L6-v2"
COLLECTION   = "ucsc_reviews"
CHROMA_DIR   = "chroma_db"
BATCH_SIZE   = 64   # chunks per embedding batch — fits in CPU RAM comfortably
DEFAULT_TOP_K = 5   # matches planning.md spec

# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------

def get_collection(chroma_dir: str = CHROMA_DIR, reset: bool = False) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_dir)
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"  Deleted existing collection '{COLLECTION}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks_path: str = "chunks.jsonl",
    chroma_dir: str = CHROMA_DIR,
    reset: bool = False,
) -> None:
    """
    Load chunks.jsonl, embed with all-MiniLM-L6-v2, upsert into ChromaDB.

    Skips chunks whose IDs already exist (safe to re-run without --reset).
    Metadata keys must be str/int/float/bool — None values are dropped.
    """
    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        sys.exit(f"chunks file not found: {chunks_path}  (run ingest.py first)")

    print(f"Loading chunks from {chunks_path}...")
    chunks = [json.loads(line) for line in chunks_file.open(encoding="utf-8") if line.strip()]
    print(f"  {len(chunks)} chunks loaded")

    collection = get_collection(chroma_dir, reset=reset)

    # Find which IDs are already in the collection
    existing_ids: set[str] = set()
    if not reset and collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])
        print(f"  {len(existing_ids)} chunks already in collection, skipping those")

    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        print("  Nothing new to embed.")
        return

    print(f"  {len(new_chunks)} chunks to embed")
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    total = len(new_chunks)
    inserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = new_chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

        ids        = [c["id"] for c in batch]
        documents  = texts

        # Build metadata dicts — ChromaDB rejects None values
        metadatas = []
        for c in batch:
            meta = {}
            for key in ("professor", "dept", "source", "rating", "professor_review_count", "class"):
                val = c.get(key)
                if val is not None:
                    meta[key] = val
            # review_text stored as metadata so generation can cite the raw review
            review_text = c.get("review_text", "")
            if review_text:
                meta["review_text"] = review_text
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        inserted += len(batch)
        print(f"  Embedded {inserted}/{total}...", end="\r")

    print(f"\n  Done. Collection '{COLLECTION}' now has {collection.count()} chunks.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_model_cache: SentenceTransformer | None = None
_collection_cache: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(MODEL_NAME)
    return _model_cache


def _get_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    global _collection_cache
    if _collection_cache is None:
        _collection_cache = get_collection(chroma_dir)
    return _collection_cache


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    professor: str | None = None,
    chroma_dir: str = CHROMA_DIR,
) -> list[dict]:
    """
    Embed query and return top-k matching chunks from ChromaDB.

    Args:
        query:      Natural-language question.
        top_k:      Number of chunks to return (default 5, per planning.md).
        professor:  Optional — filter results to a specific professor name.
                    Addresses planning.md Challenge #3 (generic-phrase retrieval
                    matching the wrong professor).
        chroma_dir: Path to ChromaDB directory.

    Returns:
        List of dicts, each with keys: id, text, score, and any metadata fields
        (professor, dept, source, rating, professor_review_count, review_text).
        Sorted by relevance (best first).
    """
    model = _get_model()
    collection = _get_collection(chroma_dir)

    query_embedding = model.encode(query).tolist()

    where_filter = None
    if professor:
        where_filter = {"professor": {"$eq": professor}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # ChromaDB returns cosine distance (0=identical, 2=opposite).
        # distance is the raw graded metric (acceptance bar: < 0.5).
        # score is a convenience similarity (1 - dist/2), range 0–1.
        distance = round(dist, 4)
        score = round(1 - dist / 2, 4)
        chunk = {"id": cid, "distance": distance, "score": score, "text": doc}
        chunk.update(meta)
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# CLI — smoke-test retrieval after embedding
# ---------------------------------------------------------------------------

def _smoke_test(chroma_dir: str) -> None:
    test_queries = [
        ("Does Edward Migliore teach his UCSC math classes in an online format?", "Edward Migliore"),
        ("Do students mention extra-credit opportunities in Scott Anderson reviews?", "Scott Anderson"),
        ("What subject does Anne Sizemore teach at UCSC?", "Anne Sizemore"),
        ("Which academic department is Ethan Miller in?", "Ethan Miller"),
        ("What department or program does A.M. Darke teach in at UCSC?", "A.M. Darke"),
    ]
    print("\nRetrieval verification — 5 eval queries, top-5 each")
    print("Acceptance bar: top-1 cosine distance < 0.5\n")
    all_pass = True
    for query, prof in test_queries:
        results = retrieve(query, top_k=5, professor=prof, chroma_dir=chroma_dir)
        if not results:
            print(f"  FAIL  {query!r} — no results returned")
            all_pass = False
            continue

        top = results[0]
        top_dist = top["distance"]
        status = "PASS" if top_dist < 0.5 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  [{status}] distance={top_dist}  Q: {query}")
        print(f"         Prof: {top.get('professor')}  Dept: {top.get('dept')}  Rating: {top.get('rating')}")
        print(f"         Top chunk: {top['text'][:180]}...")
        for r in results[1:]:
            print(f"               [{r['distance']}] {r['text'][:100]}...")
        print()

    print("All queries PASS ✓" if all_pass else "One or more queries FAILED — review retrieval before proceeding to M5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks.jsonl into ChromaDB")
    parser.add_argument("--chunks",    default="chunks.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--chroma-dir", default=CHROMA_DIR,   help="ChromaDB directory")
    parser.add_argument("--reset",     action="store_true",   help="Wipe and re-embed everything")
    parser.add_argument("--smoke-test", action="store_true",  help="Run retrieval smoke test after embedding")
    args = parser.parse_args()

    embed_chunks(args.chunks, args.chroma_dir, args.reset)

    if args.smoke_test:
        _smoke_test(args.chroma_dir)


if __name__ == "__main__":
    main()
