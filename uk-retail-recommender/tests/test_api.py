"""Tests for the FastAPI application (Phase 6)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_index_has_form():
    r = client.get("/")
    assert r.status_code == 200 and "recommend" in r.text.lower()


def test_recommend_known_customer():
    from app.main import _get_pipeline

    cid = _get_pipeline().known_customer()
    r = client.get(f"/recommend?customer_id={cid}&k=5")
    assert r.status_code == 200 and (
        "item" in r.text.lower() or "score" in r.text.lower()
    )


def test_recommend_unknown_customer_cold_start():
    r = client.get("/recommend?customer_id=-999&k=5")
    assert r.status_code == 200 and "cold" in r.text.lower()


def test_recommend_bad_input():
    r = client.get("/recommend?customer_id=abc&k=5")
    assert r.status_code == 200
