"""Optional LLM-powered query rewriter for short / ambiguous queries.

Short queries (under MIN_WORD_THRESHOLD words) get rewritten via the LLM to
expand pronouns and vague references. Longer queries pass through unchanged.
On any LLM failure the rewriter falls back to the original query — never blocks.
"""

from src.generation.llm_client import llm_invoke


REWRITE_PROMPT = """Rewrite this user question to be more specific and search-friendly, expanding any pronouns or vague references. Reply with ONLY the rewritten question, no preamble.

Original question: {question}

Rewritten question:"""

MIN_WORD_THRESHOLD = 8


def rewrite_query(question: str, llm) -> str:
    """Return a more search-friendly version of `question`, or the original on failure."""
    if len(question.split()) >= MIN_WORD_THRESHOLD:
        return question
    msgs = [("user", REWRITE_PROMPT.format(question=question))]
    try:
        resp = llm_invoke(llm, msgs)
        rewritten = resp.content.strip()
        return rewritten or question
    except Exception:  # noqa: BLE001 — never block on a rewriter failure
        return question
