from __future__ import annotations

from src.eval.langfuse_setup import get_callbacks


def test_get_callbacks_empty_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_callbacks() == []
