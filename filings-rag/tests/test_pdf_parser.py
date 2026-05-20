from pathlib import Path

import fitz

from src.ingestion.pdf_parser import parse_pdf


def _make_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for text in ("Page one body.", "Page two body."):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    out = tmp_path / "fixture.pdf"
    doc.save(out)
    doc.close()
    return out


def test_parse_yields_one_record_per_page(tmp_path):
    pdf = _make_pdf(tmp_path)
    records = list(parse_pdf(pdf, ticker="XXX", year=2024))
    assert len(records) == 2
    assert records[0]["page"] == 1
    assert records[1]["page"] == 2
    assert "Page one" in records[0]["text"]
    assert records[0]["ticker"] == "XXX"
    assert records[0]["year"] == 2024
