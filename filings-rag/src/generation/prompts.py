"""Grounded-citation prompts + citation extraction and validation."""

import re

SYSTEM_PROMPT = """You are a careful financial-filings analyst.
Answer ONLY from the provided context. For every factual claim, include a citation in the form [TICKER|YEAR|p.PAGE].
If the context does not contain the answer, reply exactly: "No relevant content found in the corpus."
Do not give investment advice."""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer (with citations):"""

CITATION_RE = re.compile(r"\[([A-Z][A-Z0-9\.\-]*)\|(\d{4})\|p\.(\d+)\]")


def extract_citations(answer: str) -> list[tuple[str, int, int]]:
    """Return all [TICKER|YEAR|p.PAGE] tuples found in `answer`."""
    return [
        (m.group(1), int(m.group(2)), int(m.group(3)))
        for m in CITATION_RE.finditer(answer)
    ]


def validate_citations(cites, retrieved_chunks) -> bool:
    """True iff every citation maps to a retrieved chunk's (ticker, year, page)."""
    if not cites:
        return False
    seen = {
        (c["metadata"]["ticker"], c["metadata"]["year"], c["metadata"]["page"])
        for c in retrieved_chunks
    }
    return all(c in seen for c in cites)
