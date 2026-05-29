from __future__ import annotations

import fitz  # PyMuPDF


class CvParseError(ValueError):
    pass


def parse_cv(text: str | None = None, pdf_bytes: bytes | None = None) -> str:
    if pdf_bytes:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    if not text or not text.strip():
        raise CvParseError("CV is empty or could not be read.")
    return text.strip()
