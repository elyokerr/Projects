"""PyMuPDF-based PDF parser that yields one record per page with metadata."""

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


def parse_pdf(path: Path, *, ticker: str, year: int) -> Iterator[dict]:
    """Yield one dict per non-empty page in the PDF.

    Each record carries `ticker`, `year`, `page` (1-indexed), and `text`.
    Empty pages (whitespace only) are skipped.
    """
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                yield {
                    "ticker": ticker,
                    "year": year,
                    "page": i,
                    "text": text,
                }
    finally:
        doc.close()
