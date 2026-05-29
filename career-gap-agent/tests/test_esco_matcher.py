import numpy as np

from src.skills.esco_matcher import EscoMatcher


class FakeIndex:
    labels = ["machine learning", "python (computer programming)", "project management"]
    uris = ["u1", "u2", "u3"]
    embeddings = np.eye(3, dtype=np.float32)


def test_exact_axis_match(monkeypatch):
    # embed() returns the basis vector for the matching label
    vec = np.array([1, 0, 0], dtype=np.float32)
    m = EscoMatcher(index=FakeIndex(), embed_fn=lambda xs: [vec for _ in xs])
    matched = m.match(["deep learning"], threshold=0.5)
    assert matched == ["machine learning"]


def test_below_threshold_dropped():
    vec = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    m = EscoMatcher(index=FakeIndex(), embed_fn=lambda xs: [vec for _ in xs])
    assert m.match(["something vague"], threshold=0.9) == []


def test_dedup_preserves_first():
    vec = np.array([1, 0, 0], dtype=np.float32)
    m = EscoMatcher(index=FakeIndex(), embed_fn=lambda xs: [vec for _ in xs])
    assert m.match(["ml", "machine learning"], threshold=0.5) == ["machine learning"]
