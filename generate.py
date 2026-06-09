"""
generate.py — Milestone 5: Grounded Generation via Groq

Takes a user question, retrieves relevant chunks from ChromaDB, and generates
a grounded answer using Groq's llama-3.3-70b-versatile — answering only from
the retrieved review context, with numbered source citations.

Usage (standalone test):
    uv run python3 generate.py

Import:
    from generate import answer
    result = answer("Does Ethan Miller curve exams?")
    print(result["answer"])
    print(result["sources"])
"""

import os
from dotenv import load_dotenv
from groq import Groq

from embed import retrieve

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_MODEL         = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE   = 0.2    # low = more factual, less creative
DEFAULT_TOP_K      = 5
DISTANCE_THRESHOLD = 0.6    # drop chunks above this distance before calling LLM

# ---------------------------------------------------------------------------
# Grounding system prompt (exact instruction — documented in README)
# ---------------------------------------------------------------------------

GROUNDING_SYSTEM_PROMPT = """You are an assistant that answers questions about UC Santa Cruz professors and courses using only student review excerpts provided below.

Rules you must follow:
1. Answer ONLY using information from the numbered review excerpts in the context. Do not use any outside knowledge about professors, universities, or courses.
2. Cite the source number(s) you used — e.g. [1], [2] — inline in your answer.
3. If the provided reviews do not contain enough information to answer the question, say exactly: "The reviews don't cover this." Do not guess or infer beyond what is written.
4. Reviews are subjective student opinions. When multiple reviews disagree, summarize both perspectives fairly — do not present one opinion as consensus fact.
5. Keep your answer concise (3–5 sentences unless the question requires more detail).
"""

# ---------------------------------------------------------------------------
# Professor detection (Challenge #3 mitigation: scope retrieval by name)
# ---------------------------------------------------------------------------

KNOWN_PROFESSORS = [
    "Ethan Miller",
    "A.M. Darke",
    "Ryan Coonerty",
    "Jesse Kass",
    "Anne Sizemore",
    "Scott Anderson",
    "Edward Migliore",
    "Steven Owen",
]

def _detect_professor(query: str) -> str | None:
    """
    Return the first known professor name found (case-insensitive) in the query.
    Used to scope ChromaDB retrieval by metadata filter so generic phrases
    ("hard exams", "extra credit") don't match the wrong professor.
    """
    q_lower = query.lower()
    for prof in KNOWN_PROFESSORS:
        # Match on last name or full name
        last_name = prof.split()[-1].lower()
        if last_name in q_lower or prof.lower() in q_lower:
            return prof
    return None


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """
    Build the numbered context block injected into the LLM prompt.

    Each entry includes a metadata header so the model knows who the review
    is about — critical for grounding when the review text uses pronouns.

    Example:
        [1] Edward Migliore (Mathematics) | Rating: 2.0 | 263 total reviews
        Edward Migliore — Mathematics
        I expected him to accommodate his students despite this course being
        taught online...
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        prof  = chunk.get("professor", "Unknown")
        dept  = chunk.get("dept", "")
        rating = chunk.get("rating")
        count  = chunk.get("professor_review_count")

        header_parts = [f"{prof} ({dept})" if dept else prof]
        if rating is not None:
            header_parts.append(f"Rating: {rating}")
        if count is not None:
            header_parts.append(f"{count} total reviews")

        lines.append(f"[{i}] {' | '.join(header_parts)}")
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Groq client (lazy init)
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Main answer function
# ---------------------------------------------------------------------------

def answer(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Full RAG pipeline for one question:
      1. Detect professor (if named) → scope retrieval
      2. Retrieve top-k chunks from ChromaDB
      3. Filter low-relevance chunks (distance > DISTANCE_THRESHOLD)
      4. If nothing survives filter → return refusal without calling LLM
      5. Format context → call Groq → return grounded answer + sources

    Returns:
        {
          "answer":  str,          # grounded answer text with [n] citations
          "sources": list[dict],   # deduped source metadata shown to user
          "chunks_used": int,      # how many chunks were sent to the LLM
        }
    """
    professor_filter = _detect_professor(query)
    chunks = retrieve(query, top_k=top_k, professor=professor_filter)

    # Filter by relevance distance
    relevant = [c for c in chunks if c.get("distance", 1.0) <= DISTANCE_THRESHOLD]

    if not relevant:
        return {
            "answer": "The reviews don't cover this. No sufficiently relevant reviews were found for your question.",
            "sources": [],
            "chunks_used": 0,
        }

    context_text = format_context(relevant)

    user_message = f"""Context (student reviews):
{context_text}

Question: {query}"""

    client = _get_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        messages=[
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    answer_text = completion.choices[0].message.content.strip()

    # Append references only when the LLM gave a substantive answer.
    # If it refused ("The reviews don't cover this"), references add no value.
    is_refusal = "don't cover this" in answer_text.lower() or "reviews don't" in answer_text.lower()
    if not is_refusal:
        ref_lines = ["**References:**"]
        for i, chunk in enumerate(relevant, 1):
            prof   = chunk.get("professor", "Unknown")
            dept   = chunk.get("dept", "")
            rating = chunk.get("rating")
            url    = chunk.get("source", "")
            review = chunk.get("review_text", chunk.get("text", ""))
            label  = f"{prof} ({dept})" if dept else prof
            if rating is not None:
                label += f", rating {rating}"
            excerpt = review[:100].strip()
            if len(review) > 100:
                excerpt += "..."
            source_line = f"[{i}] {label} — {url}" if url else f"[{i}] {label}"
            ref_lines.append(f'{source_line}  \n> "{excerpt}"')
        answer_text = answer_text + "\n\n" + "\n\n".join(ref_lines)

    # Build deduped sources list from chunks sent to the LLM
    seen: set[str] = set()
    sources: list[dict] = []
    for chunk in relevant:
        prof = chunk.get("professor", "")
        key = prof or chunk.get("id", "")
        if key not in seen:
            seen.add(key)
            sources.append({
                "professor": prof,
                "dept":      chunk.get("dept", ""),
                "rating":    chunk.get("rating"),
                "source":    chunk.get("source", ""),
                "review_count": chunk.get("professor_review_count"),
            })

    return {
        "answer":      answer_text,
        "sources":     sources,
        "chunks_used": len(relevant),
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "Does Edward Migliore teach his UCSC math classes in an online format?",
        "Does Scott Anderson offer extra credit in his classes?",
        "What subject does Anne Sizemore teach at UCSC?",
        "What's the UCSC tuition?",   # off-corpus refusal test
    ]

    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"Q: {q}")
        result = answer(q)
        print(f"\nAnswer ({result['chunks_used']} chunks used):")
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  - {s['professor']} ({s['dept']}) | rating {s['rating']} | {s['review_count']} reviews")
                print(f"    {s['source']}")
