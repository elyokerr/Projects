import pytest

from src.generation.llm_client import build_chat_model, simple_complete


def test_raises_when_no_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_chat_model()


def test_simple_complete_uses_injected_model():
    class FakeMsg:
        content = '["python"]'

    class FakeModel:
        def invoke(self, prompt):
            return FakeMsg()

    assert simple_complete("anything", model=FakeModel()) == '["python"]'
