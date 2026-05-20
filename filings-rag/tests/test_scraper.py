from pathlib import Path
import pytest
from src.ingestion.scraper import download_pdf, validate_pdf_url


def test_validate_rejects_non_pdf_content_type(monkeypatch):
    class Resp:
        headers = {"Content-Type": "text/html"}
        status_code = 200
    monkeypatch.setattr("src.ingestion.scraper._head", lambda url: Resp())
    assert validate_pdf_url("https://example.com/not.pdf") is False


def test_validate_accepts_pdf_content_type(monkeypatch):
    class Resp:
        headers = {"Content-Type": "application/pdf"}
        status_code = 200
    monkeypatch.setattr("src.ingestion.scraper._head", lambda url: Resp())
    assert validate_pdf_url("https://example.com/ok.pdf") is True


def test_download_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.scraper._get_bytes", lambda url: b"%PDF-1.4 fake")
    out = tmp_path / "AZN" / "2024.pdf"
    download_pdf("https://example.com/ok.pdf", out, delay_s=0)
    assert out.read_bytes().startswith(b"%PDF-")
