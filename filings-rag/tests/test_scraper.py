from pathlib import Path
import pytest
from src.ingestion.scraper import download_pdf, validate_pdf_url, is_pdf_bytes


def test_validate_rejects_non_pdf_url_with_html_content_type(monkeypatch):
    """URL doesn't end in .pdf and HEAD says text/html -> reject."""
    class Resp:
        headers = {"Content-Type": "text/html"}
        status_code = 200
    monkeypatch.setattr("src.ingestion.scraper._head", lambda url: Resp())
    assert validate_pdf_url("https://example.com/something") is False


def test_validate_accepts_pdf_extension_regardless_of_head():
    """URL ending in .pdf is trusted (magic bytes verified post-download)."""
    assert validate_pdf_url("https://example.com/report.pdf") is True


def test_validate_accepts_pdf_content_type(monkeypatch):
    """Non-.pdf URL with application/pdf Content-Type is accepted."""
    class Resp:
        headers = {"Content-Type": "application/pdf"}
        status_code = 200
    monkeypatch.setattr("src.ingestion.scraper._head", lambda url: Resp())
    assert validate_pdf_url("https://example.com/download") is True


def test_is_pdf_bytes():
    assert is_pdf_bytes(b"%PDF-1.4 fake") is True
    assert is_pdf_bytes(b"<html>") is False
    assert is_pdf_bytes(b"") is False


def test_download_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.scraper._get_bytes", lambda url: b"%PDF-1.4 fake content")
    out = tmp_path / "AZN" / "2024.pdf"
    download_pdf("https://example.com/ok.pdf", out, delay_s=0)
    assert out.read_bytes().startswith(b"%PDF-")


def test_download_rejects_non_pdf_bytes(tmp_path, monkeypatch):
    """If the response doesn't start with %PDF- magic, download raises."""
    monkeypatch.setattr("src.ingestion.scraper._get_bytes", lambda url: b"<html>oops</html>")
    out = tmp_path / "X" / "2024.pdf"
    with pytest.raises(ValueError, match="did not start with %PDF"):
        download_pdf("https://example.com/blocked.pdf", out, delay_s=0)
    assert not out.exists()
