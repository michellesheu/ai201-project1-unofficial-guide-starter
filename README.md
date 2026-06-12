# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section _after_ you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

Student experiences with specific professors and courses at UC Santa Cruz — what teaching style, workload, grading, and exam difficulty are actually like in each class, and which instructors students recommend or avoid. This is hard to find through official channels because the registrar's catalog and the Student Experience of Teaching Survey (SET) summaries describe course content and aggregate numbers, not the candid, instructor-specific advice students trade informally. The real signal lives scattered across Rate My Professors reviews, the student-built Slugtistics GPA/review site, and r/UCSC threads — sources no single official page aggregates.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| #   | Source                               | Type                          | URL or file path                                                           |
| --- | ------------------------------------ | ----------------------------- | -------------------------------------------------------------------------- |
| 1   | RMP — UCSC school hub                | Review aggregator             | https://www.ratemyprofessors.com/school/1078                               |
| 2   | RMP — Ethan Miller (CS)              | Professor reviews             | https://www.ratemyprofessors.com/professor/136264                          |
| 3   | RMP — A.M. Darke (Art/Games)         | Professor reviews             | https://www.ratemyprofessors.com/professor/2463375                         |
| 4   | RMP — Ryan Coonerty (Politics)       | Professor reviews             | https://www.ratemyprofessors.com/professor/439427                          |
| 5   | RMP — Jesse Kass (Math)              | Professor reviews             | https://www.ratemyprofessors.com/professor/2895784                         |
| 6   | RMP — Anne Sizemore (Chem)           | Professor reviews             | https://www.ratemyprofessors.com/professor/2989548                         |
| 7   | RMP — Scott Anderson                 | Professor reviews             | https://www.ratemyprofessors.com/professor/2319462                         |
| 8   | RMP — Edward Migliore (Math, online) | Professor reviews             | https://www.ratemyprofessors.com/professor/218123                          |
| 9   | RMP — Steven Owen                    | Professor reviews             | https://www.ratemyprofessors.com/professor/2346100                         |
| 10  | Slugtistics                          | GPA data + instructor reviews | https://slugtistics.com/about                                              |
| 11  | UCSC IRAPS grade data                | Official grade distributions  | https://iraps.ucsc.edu/campus-data/student-data/grades-by-course-and-term/ |
| 12  | r/UCSC subreddit                     | Community discussion threads  | https://www.reddit.com/r/UCSC/                                             |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** One chunk per individual review, targeting ~400–600 characters. Reviews that exceed 2000 characters (approximately 512 tokens) are split into fixed-size windows under a fallback path.

**Overlap:** No overlap on per-review chunks — each review is an atomic unit. The fallback windowing path carries ~50 characters (roughly one sentence) of overlap from one window into the next, so that the boundary between two windows doesn't cut off a mid-sentence thought.

**Why these choices fit your documents:** RMP reviews are naturally bounded, self-contained opinions — typically 1–3 sentences, 50–150 words. Splitting at review boundaries rather than at fixed token counts keeps each chunk attributable to exactly one professor and one rating, which is critical for grounded attribution. A fixed-size split across concatenated reviews would silently merge two unrelated opinions about different professors into one chunk, corrupting the embedding and making retrieval misleading. Preprocessing before chunking strips HTML tags, unescapes HTML entities, removes nav boilerplate lines (e.g. "Sign In", "Helpful? Yes (12)"), and drops exact duplicate reviews using an MD5 hash of normalized text.

**Final chunk count:** 580 chunks from 587 loaded records (7 exact duplicates dropped).

**Sample chunks (5 labeled examples):**

> **Chunk 1** — Edward Migliore | Mathematics | Rating: 2.0 | Source: Rate My Professors
> ```
> Edward Migliore — Mathematics
> I expected him to accommodate his students despite this course being taught online but all of the
> lectures are just recorded classes from 2019 back when classes were LIVE. it's been tough trying
> to engage in the material for this class because lectures r 3 times a week
> ```

> **Chunk 2** — Scott Anderson | Accounting | Rating: 3.0 | Source: Rate My Professors
> ```
> Scott Anderson — Accounting
> Anderson is a okay professor. He is not that interesting as he mainly just reads off the slides.
> 2 midterms that were online and based off of home works and pre-readings. He does give out extra
> credit for answering questions in class. Case study project.
> ```

> **Chunk 3** — Anne Sizemore | Chemistry | Rating: 5.0 | Source: Rate My Professors
> ```
> Anne Sizemore — Chemistry
> Dr. Sizemore is possibly hands down my favorite professor at UCSC. The lectures were amazing which
> motivated me to go and exams were extremely straightforward. The way she teaches is super effective
> and she held true to her word that "everyone can learn Ochem".
> ```

> **Chunk 4** — Ethan Miller | Computer Science | Rating: 2.0 | Source: Rate My Professors
> ```
> Ethan Miller — Computer Science
> A stain on the computer science department at UCSC. This guy is the embodiment of everything wrong
> with professors at UCSC. He's lazy, takes other professors' assignments (can barely explain them),
> and thinks all students are cheaters.
> ```

> **Chunk 5** — Jesse Kass | Mathematics | Rating: 5.0 | Source: Rate My Professors
> ```
> Jesse Kass — Mathematics
> Love love love this prof. Went to EVERY lectures and learned so so much
> ```

Each chunk begins with a `Professor — Department` self-containment header so retrieved chunks are interpretable without the surrounding document context.

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. It produces 384-dimensional embeddings with a 256-token context window, runs entirely locally with no API cost, and loads fast enough for interactive queries. The 256-token limit is not a problem for this corpus because individual RMP reviews are almost always under 100 tokens.

**Production tradeoff reflection:** Without cost constraints, the most important tradeoffs to weigh would be context length and domain specificity. `all-MiniLM-L6-v2` truncates at 256 tokens, which would silently lose information if the corpus were extended to include longer Reddit posts or multi-paragraph course descriptions. A model with a longer context window — such as `bge-large-en` (512 tokens) or an API-hosted model like OpenAI's `text-embedding-3-large` (8191 tokens) — would handle longer documents without truncation. On domain specificity: general-purpose embedding models may underweight UCSC-specific slang, professor nicknames, or course shorthand ("BChem", "CMPS 12A") that doesn't appear in the model's training data, leading to weaker semantic matches. A model fine-tuned on student review text would retrieve more precisely in those edge cases. Finally, local models like MiniLM avoid per-query API cost and latency but require managing the model on disk; API-hosted embeddings trade those concerns for better accuracy and zero infrastructure overhead.

---

## Retrieval

For each evaluation query, the system retrieved 5 chunks filtered to the named professor via ChromaDB metadata (`where: {professor: $eq}`). The table below shows the top returned chunk text and why it is relevant.

| # | Query | Top returned chunk (truncated) | Why relevant | Top-1 distance |
|---|-------|-------------------------------|--------------|----------------|
| 1 | Does Edward Migliore teach his UCSC math classes in an online format? | "I expected him to accommodate his students despite this course being taught online but all of the lectures are just recorded classes from 2019..." | Directly states the class is taught online and explains the pre-recorded lecture format — the exact subject of the query. Distance 0.241 places it well within the acceptance threshold. | 0.241 ✓ |
| 2 | Do students mention extra-credit opportunities in Scott Anderson's reviews? | "Anderson is a okay professor... He does give out extra credit for answering questions in class." | Contains the phrase "extra credit" in context that matches the query semantically; the review describes the specific mechanism (answering in-class questions), not just that extra credit exists. | 0.488 ✓ |
| 3 | What subject does Anne Sizemore teach at UCSC? | "Dr. Sizemore is possibly hands down my favorite professor at UCSC... she held true to her word that 'everyone can learn Ochem'." | Names the subject "Ochem" (organic chemistry) explicitly; the self-containment header also carries "Chemistry" as the department label, reinforcing the match. | 0.303 ✓ |
| 4 | Which academic department is Ethan Miller associated with at UCSC? | "A stain on the computer science department at UCSC." | Names "computer science department" verbatim — exact lexical and semantic match to the query asking for department. The phrase appears in the first sentence of the chunk. | 0.224 ✓ |
| 5 | What department or program does A.M. Darke teach in at UCSC? | "UCSC just hires anybody to be a professor I guess now. Very unqualified and unprofessional professor..." | The chunk carries the metadata department "Art" in its self-containment header, which is why it ranked highest — but the review text itself does not name the full program ("Art & Games / Performance, Play & Design"). This is the root cause of the partial-accuracy failure in Q5. | 0.351 ✓ |

All top-1 distances are below the 0.5 acceptance threshold. Retrieval is scoped by professor name via ChromaDB `where` filter before semantic ranking, which prevents generic phrases ("hard exams", "extra credit") from matching the wrong professor (Challenge #3 from planning.md).

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:** The system prompt given to `llama-3.3-70b-versatile` is:

> You are an assistant that answers questions about UC Santa Cruz professors and courses using only student review excerpts provided below.
>
> Rules you must follow:
>
> 1. Answer ONLY using information from the numbered review excerpts in the context. Do not use any outside knowledge about professors, universities, or courses.
> 2. Cite the source number(s) you used — e.g. [1], [2] — inline in your answer.
> 3. If the provided reviews do not contain enough information to answer the question, say exactly: "The reviews don't cover this." Do not guess or infer beyond what is written.
> 4. Reviews are subjective student opinions. When multiple reviews disagree, summarize both perspectives fairly — do not present one opinion as consensus fact.
> 5. Keep your answer concise (3–5 sentences unless the question requires more detail).

In addition to the system prompt, grounding is reinforced structurally: each chunk is numbered `[1]` through `[n]` in the context block and prefixed with a metadata header (`Professor (Dept) | Rating: X.X | N total reviews`) before the review text. This makes it unambiguous which chunk corresponds to each citation and prevents the model from treating one review as universal. Chunks with a cosine distance above 0.6 are filtered out before the LLM call — if no chunks survive the filter, the system returns a refusal without calling the LLM at all.

**How source attribution is surfaced in the response:** The LLM is instructed to cite `[n]` inline. After generation, a numbered **References** block is appended programmatically (not by the LLM) mapping each `[n]` to the professor name, department, star rating, source URL, and a 100-character excerpt from the review text. This means every citation in the answer can be traced to a specific review, not just a professor page. If the answer is a refusal ("The reviews don't cover this"), the references block is suppressed.

**Example output — grounded answer with visible source citations:**

> **Question:** Does Scott Anderson offer extra credit in his classes?
>
> **System response:**
> Yes, Scott Anderson offers extra credit in his classes [1, 4]. According to one review, he gives out extra credit for answering questions in class [1], and another review mentions an opportunity for extra credit [4].
>
> **References:**
>
> [1] Scott Anderson (Accounting), rating 3.0 — https://www.ratemyprofessors.com/professor/2319462
> > "Anderson is a okay professor. He is not that interesting as he mainly just reads off the slides. 2 m..."
>
> [2] Scott Anderson (Accounting), rating 5.0 — https://www.ratemyprofessors.com/professor/2319462
> > "The class is difficult but if you study hard and make an effort to understand the concepts you will..."
>
> [3] Scott Anderson (Accounting), rating 4.0 — https://www.ratemyprofessors.com/professor/2319462
> > "I took Econ 10A with Scott. He is such a great teacher who is very knowledgeable as he works as a CP..."
>
> [4] Scott Anderson (Accounting), rating 2.0 — https://www.ratemyprofessors.com/professor/2319462
> > "Mr. Anderson is an awesome guy. Enjoyed his class, opportunity for extra credit, However, he doesn't..."
>
> [5] Scott Anderson (Accounting), rating 4.0 — https://www.ratemyprofessors.com/professor/2319462
> > "A solid choice for accounting. Definitely knows his stuff and is resonable when it comes to tests. T..."

**Example output — out-of-scope query with explicit refusal:**

> **Question:** What's the UCSC tuition?
>
> **System response:**
> The reviews don't cover this. No sufficiently relevant reviews were found for your question.

The tuition query triggers the distance filter: no retrieved chunk had a cosine distance ≤ 0.6 to the query after professor-scoped filtering was skipped (no professor name detected), so the system returns a refusal without calling the LLM. This confirms the grounding mechanism blocks out-of-corpus questions before generation.

---

## Query Interface

The web interface is built with Gradio (`app.py`) and runs locally via `uv run python3 app.py`. It exposes three components:

- **Question textbox** — free-text input with a placeholder prompt ("e.g. Does Ethan Miller curve his exams?") and a primary "Ask" button. Submitting via Enter key also triggers the pipeline.
- **Answer panel** — Markdown output showing the grounded answer with inline `[n]` citations and a programmatically appended **References** block linking each citation to the source URL and a 100-character review excerpt.
- **Sources panel** — Markdown list of deduplicated professor entries (name, department, star rating, review count, RMP URL) drawn from whichever chunks were sent to the LLM.
- **Example questions** — six clickable preset queries that populate the textbox: covers online format, extra credit, subject, overall quality, exam style, and an out-of-scope refusal test.
- **Professors in database** — footer line listing all 8 professors whose reviews are indexed, so users know what the system can answer.

**Sample interaction transcript:**

```
User: Does Edward Migliore teach his UCSC math classes in an online format?

Answer (5 chunks used):
According to the reviews, Edward Migliore's math class is taught in an online format [1], [3], [4],
[5]. However, one review mentions that the lectures are recorded from 2019 when classes were live [1],
suggesting the online format may be a result of past circumstances. Another review notes that the
class is self-paced [5].

References:
[1] Edward Migliore (Mathematics), rating 2.0 — https://www.ratemyprofessors.com/professor/218123
    "I expected him to accommodate his students despite this course being taught online but all of the le..."
[2] Edward Migliore (Mathematics), rating 4.0 — https://www.ratemyprofessors.com/professor/218123
    "Migliore is a very solid choice when choosing a math professor at UCSC, his lectures are clear and c..."
[3] Edward Migliore (Mathematics), rating 5.0 — https://www.ratemyprofessors.com/professor/218123
    "Always take online math, really easy class if you do the homework on time. The quizzes are the same..."
[4] Edward Migliore (Mathematics), rating 4.0 — https://www.ratemyprofessors.com/professor/218123
    "Professor Migliore was involved in the class only as the voice of pre-recorded lectures, in which he..."
[5] Edward Migliore (Mathematics), rating 3.0 — https://www.ratemyprofessors.com/professor/218123
    "This online math class was self-paced, with predictable assessments. Staying on top of the work is k..."

Sources:
- Edward Migliore — Mathematics | Rating: 2.0 | 263 reviews
  https://www.ratemyprofessors.com/professor/218123
```

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| #   | Question                                                                    | Expected answer                                                      | System response (summarized)                                                                                                                                             | Retrieval quality | Response accuracy  |
| --- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------ |
| 1   | Does Edward Migliore teach his UCSC math classes in an online format?       | Yes — RMP reviews describe online/asynchronous instruction.          | Yes, confirmed across 4 of 5 retrieved chunks — pre-recorded lectures, self-paced format, explicitly "taught online." Noted that recordings date from 2019 live classes. | Relevant          | Accurate           |
| 2   | Do students mention extra-credit opportunities in Scott Anderson's reviews? | Yes — reviews reference extra credit as part of his grading.         | Yes — two reviews cited: one mentions extra credit for answering questions in class, another notes "opportunity for extra credit."                                       | Relevant          | Accurate           |
| 3   | What subject does Anne Sizemore teach at UCSC?                              | Organic Chemistry (Chemistry department).                            | Chemistry; specifically organic chemistry, named in two of five retrieved reviews.                                                                                       | Relevant          | Accurate           |
| 4   | Which academic department is Ethan Miller associated with at UCSC?          | Computer Science.                                                    | Computer Science — stated directly in the top retrieved chunk ("a stain on the computer science department").                                                            | Relevant          | Accurate           |
| 5   | What department or program does A.M. Darke teach in at UCSC?                | Art & Games / Performance, Play & Design (within the Arts division). | "Art department" — the system returned the RMP department label, which is "Art", not the full UCSC program name "Art & Games / Performance, Play & Design."              | Relevant          | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** What department or program does A.M. Darke teach in at UCSC?

**What the system returned:** "A.M. Darke teaches in the Art department at UCSC." The system answered confidently and cited five relevant chunks — but the answer is incomplete. The expected answer is "Art & Games / Performance, Play & Design (within the Arts division)," which is the actual UCSC program name.

**Root cause (tied to a specific pipeline stage):** The failure occurs at the **ingestion stage** — specifically in how `scrape.py` stores the `department` field. RMP's GraphQL API returns a coarse taxonomy label ("Art") rather than the formal UCSC program name. The scraper stored that string verbatim, so every A.M. Darke chunk has `"dept": "Art"` in its metadata and embedded text. The embedding model never saw the program name "Art & Games," so it could not retrieve or surface it regardless of how the query was phrased. The model answered faithfully from its context — the context itself was wrong.

**What you would change to fix it:** Augment the scraper to cross-reference the RMP professor page against UCSC's official faculty directory or the UCSC course catalog, which carries the precise department/program name. Alternatively, manually correct department metadata for the affected professors in the JSONL source files before re-running `ingest.py` and `embed.py --reset`.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:** The Chunking Strategy section's decision to chunk at review boundaries rather than fixed token counts directly shaped the most important design choice in `ingest.py`. When writing the chunker, it would have been easy to default to a simple 512-token sliding window (which is what most chunking tutorials show). The spec's reasoning — that RMP reviews are atomic, self-contained opinions and that merging two reviews into one chunk would silently corrupt embeddings by conflating two professors — gave a concrete justification for implementing the more complex per-review splitter. Without that written rationale, the shortcut would have been tempting and the retrieval quality would have been worse.

**One way your implementation diverged from the spec, and why:** The spec listed Slugtistics and the UCSC IRAPS grade data as document sources (rows 10 and 11 of the Documents table). Neither made it into the final corpus. Slugtistics is a JavaScript-heavy SPA with no accessible static content, and IRAPS serves structured data tables that are difficult to chunk meaningfully as review text. Rather than include low-quality or structurally incompatible documents just to hit a number, the implementation focused entirely on RMP professor reviews, which are the richest and most query-relevant source. The Reddit source (r/UCSC) was also attempted but blocked by Reddit's 403 API restrictions on unauthenticated search requests. The divergence was pragmatic. The spec was written before scraping revealed which sources were actually accessible.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- _What I gave the AI:_ The Chunking Strategy section from `planning.md` (chunk size, overlap, reasoning, preprocessing steps) and a description of the input format — JSONL files where each line is one professor review with fields `text`, `professor`, `dept`, `source`, `rating`.
- _What it produced:_ A complete `ingest.py` with `load_records()`, `clean_text()`, `chunk_records()`, and `write_jsonl()`. The initial draft used `convert_to_list=True` as a `model.encode()` keyword argument, which raised a `ValueError` on the installed version of `sentence-transformers` (the parameter was removed). It also used Unicode box-drawing characters (`──────────────`) as visual separators in chunk text, which don't affect semantics but added noise.
- _What I changed or overrode:_ Fixed `model.encode(..., convert_to_list=True)` → `model.encode(...).tolist()`. Removed the separator lines. Directed the AI to add a self-containment header (`"{professor} — {dept}\n{review}"`) to every chunk text after identifying that bare reviews like "he grades hard" would have no named subject when retrieved — a usability problem that wasn't anticipated in the original spec.

**Instance 2**

- _What I gave the AI:_ The Retrieval Approach section from `planning.md` (embedding model, top-k=5, metadata filter for professor name), plus the pipeline diagram showing ChromaDB as the vector store, and the already-written `ingest.py` as context for the output format.
- _What it produced:_ `embed.py` with `embed_chunks()`, `retrieve()`, a lazy-init model cache, and a `--smoke-test` CLI path. The initial `retrieve()` returned only a converted similarity score (`1 - dist/2`) and discarded the raw cosine distance, making it impossible to verify the acceptance criterion (top-1 distance < 0.5) directly from the output.
- _What I changed or overrode:_ Directed the AI to expose the raw `distance` field alongside `score` in each returned chunk dict, and to add a per-query PASS/FAIL verdict in the smoke test. Also changed the smoke test from 3 generic queries to all 5 eval plan questions with top-5 results printed, so the verification was directly tied to the graded spec rather than ad-hoc.
