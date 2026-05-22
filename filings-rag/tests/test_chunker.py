from src.ingestion.chunker import chunk_pages


def test_chunks_preserve_page_metadata():
    pages = [
        {"ticker": "X", "year": 2024, "page": 1, "text": "Section one. " * 200},
        {"ticker": "X", "year": 2024, "page": 2, "text": "Section two. " * 200},
    ]
    chunks = list(chunk_pages(pages, max_tokens=200, overlap=20))
    assert all("ticker" in c and "page" in c for c in chunks)
    tickers = {c["ticker"] for c in chunks}
    assert tickers == {"X"}


def test_chunk_text_under_token_limit():
    pages = [{"ticker": "X", "year": 2024, "page": 1, "text": "word " * 1000}]
    chunks = list(chunk_pages(pages, max_tokens=128, overlap=10))
    for c in chunks:
        assert len(c["text"].split()) <= 128
