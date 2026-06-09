"""
app.py — Milestone 5: Gradio web interface for the UCSC RAG pipeline

Run:
    uv run python3 app.py

Opens a local browser UI. Ask any question about UCSC professors —
answers are grounded strictly in student reviews with source citations.
"""

import gradio as gr
from generate import answer, KNOWN_PROFESSORS

# ---------------------------------------------------------------------------
# UI logic
# ---------------------------------------------------------------------------

def ask(query: str) -> tuple[str, str]:
    """
    Called on each submission. Returns (answer_md, sources_md) for the two
    Gradio Markdown output components.
    """
    query = query.strip()
    if not query:
        return "Please enter a question.", ""

    result = answer(query)

    answer_md = result["answer"]

    if not result["sources"]:
        sources_md = "_No sources retrieved._"
    else:
        lines = []
        for s in result["sources"]:
            prof  = s.get("professor") or "Unknown"
            dept  = s.get("dept") or ""
            rating = s.get("rating")
            count  = s.get("review_count")
            url    = s.get("source") or ""

            label = f"**{prof}**"
            if dept:
                label += f" — {dept}"
            if rating is not None:
                label += f" | Rating: {rating}"
            if count is not None:
                label += f" | {count} reviews"
            if url:
                label += f"\n  [{url}]({url})"
            lines.append(f"- {label}")
        sources_md = "\n".join(lines)

    return answer_md, sources_md


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

example_questions = [
    "Does Edward Migliore teach online?",
    "Does Scott Anderson offer extra credit?",
    "What subject does Anne Sizemore teach?",
    "Is Ethan Miller a good professor?",
    "What do students say about Ryan Coonerty's exams?",
    "What's the UCSC tuition?",
]

with gr.Blocks(title="UCSC Unofficial Guide") as demo:
    gr.Markdown(
        """# 🐌 UCSC Unofficial Guide
        Ask questions about UC Santa Cruz professors based on student reviews.
        Answers are grounded strictly in scraped Rate My Professors data — no outside knowledge.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. Does Ethan Miller curve his exams?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")

    gr.Markdown("### Answer")
    answer_out = gr.Markdown(value="_Ask a question above._")

    gr.Markdown("### Sources")
    sources_out = gr.Markdown(value="")

    gr.Markdown(
        f"**Professors in the database:** {', '.join(KNOWN_PROFESSORS)}"
    )

    gr.Examples(
        examples=example_questions,
        inputs=question_box,
        label="Example questions",
    )

    submit_btn.click(fn=ask, inputs=question_box, outputs=[answer_out, sources_out])
    question_box.submit(fn=ask, inputs=question_box, outputs=[answer_out, sources_out])

if __name__ == "__main__":
    demo.launch()
