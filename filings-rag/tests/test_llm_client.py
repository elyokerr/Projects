import pytest

from src.generation.llm_client import build_llm, ProviderError


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ProviderError):
        build_llm()


def test_groq_default_pulls_key_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    llm = build_llm()
    assert "llama-3.3-70b" in llm.model.lower() or "llama" in llm.model.lower()
