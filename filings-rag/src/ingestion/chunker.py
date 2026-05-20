"""Page-aware chunker that produces overlapping windows with stable hash IDs."""

from typing import Iterable, Iterator
import hashlib


def chunk_pages(
    pages: Iterable[dict],
    max_tokens: int = 512,
    overlap: int = 50,
) -> Iterator[dict]:
    """Slice each page's text into overlapping windows of <= max_tokens whitespace tokens.

    Chunks never span pages (preserving page metadata for citations). Each chunk
    carries the source `ticker`, `year`, `page`, plus a stable 16-char SHA-256
    `chunk_hash` of its text.
    """
    for page in pages:
        tokens = page["text"].split()
        if not tokens:
            continue
        step = max_tokens - overlap
        for start in range(0, len(tokens), step):
            window = tokens[start:start + max_tokens]
            if not window:
                break
            text = " ".join(window)
            yield {
                "text": text,
                "ticker": page["ticker"],
                "year": page["year"],
                "page": page["page"],
                "chunk_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            }
            if start + max_tokens >= len(tokens):
                break
