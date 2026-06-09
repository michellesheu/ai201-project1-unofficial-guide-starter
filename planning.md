# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Student experiences with specific professors and courses at UC Santa Cruz — what teaching style, workload, grading, and exam difficulty are actually like in each class, and which instructors students recommend or avoid. This is hard to find through official channels because the registrar's catalog and the Student Experience of Teaching Survey (SET) summaries describe course content and aggregate numbers, not the candid, instructor-specific advice students trade informally. The real signal lives scattered across Rate My Professors reviews, the student-built Slugtistics GPA/review site, and r/UCSC threads — sources no single official page aggregates.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | RMP — UCSC school hub | Aggregated ratings for all UCSC professors | https://www.ratemyprofessors.com/school/1078 |
| 2 | RMP — Ethan Miller (CS) | Reviews for a high-profile CS prof with polarized opinions | https://www.ratemyprofessors.com/professor/136264 |
| 3 | RMP — A.M. Darke (Art/Games) | Reviews covering an Art & Games dept professor | https://www.ratemyprofessors.com/professor/2463375 |
| 4 | RMP — Ryan Coonerty (Politics) | Reviews for a Politics/Community Studies prof | https://www.ratemyprofessors.com/professor/439427 |
| 5 | RMP — Jesse Kass (Math) | Reviews for a well-regarded Math dept professor | https://www.ratemyprofessors.com/professor/2895784 |
| 6 | RMP — Anne Sizemore (Chem) | Reviews for an Organic Chemistry professor | https://www.ratemyprofessors.com/professor/2989548 |
| 7 | RMP — Scott Anderson | Reviews covering extra credit and grading style | https://www.ratemyprofessors.com/professor/2319462 |
| 8 | RMP — Edward Migliore (Math, online) | Reviews for an online Math instructor | https://www.ratemyprofessors.com/professor/218123 |
| 9 | RMP — Steven Owen | Reviews for a UCSC professor (diverse dept coverage) | https://www.ratemyprofessors.com/professor/2346100 |
| 10 | Slugtistics | Student-built UCSC class search with GPA data and instructor reviews | https://slugtistics.com/about |
| 11 | UCSC IRAPS grade data | Official grade distributions by course and term | https://iraps.ucsc.edu/campus-data/student-data/grades-by-course-and-term/ |
| 12 | r/UCSC subreddit | Community threads on professor recommendations and easy GEs | https://www.reddit.com/r/UCSC/ |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**

**Overlap:**

**Reasoning:**

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**

**Top-k:**

**Production tradeoff reflection:**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.

2.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
