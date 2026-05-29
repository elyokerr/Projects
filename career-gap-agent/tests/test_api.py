"""Tests for the FastAPI web app (Phase 5).

These tests run with NO secrets — the empty-CV path returns before the LLM
step, and health/index require no external services.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_index_serves_form():
    r = client.get("/")
    assert r.status_code == 200 and "career" in r.text.lower()


def test_analyze_empty_cv_returns_message():
    r = client.post(
        "/analyze",
        data={"cv_text": "", "role": "data scientist", "location": "london"},
    )
    assert r.status_code == 200 and "cv" in r.text.lower()
