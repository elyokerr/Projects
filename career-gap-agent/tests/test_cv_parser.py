import pytest

from src.data.cv_parser import CvParseError, parse_cv


def test_plain_text_passthrough():
    assert "python" in parse_cv(text="I know Python and SQL.").lower()


def test_empty_raises():
    with pytest.raises(CvParseError):
        parse_cv(text="   ")


def test_pdf_bytes(tmp_path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Experienced in PyTorch and Docker.")
    pdf_bytes = doc.tobytes()
    out = parse_cv(pdf_bytes=pdf_bytes)
    assert "pytorch" in out.lower()
