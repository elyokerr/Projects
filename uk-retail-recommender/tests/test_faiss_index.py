import numpy as np

from src.retrieval.faiss_index import FaissIndex


def test_build_and_query_returns_nearest():
    vecs = np.array([[1,0,0],[0,1,0],[0.9,0.1,0]], dtype="float32")
    idx = FaissIndex(dim=3).build(vecs, ids=["a","b","c"])
    hits = idx.query(np.array([1,0,0], dtype="float32"), k=2)
    assert hits[0][0] == "a"  # nearest by inner product on normalised vecs
    assert "c" in [h[0] for h in hits]


def test_query_caps_k_to_index_size():
    vecs = np.array([[1,0,0],[0,1,0]], dtype="float32")
    idx = FaissIndex(dim=3).build(vecs, ids=["a","b"])
    hits = idx.query(np.array([1,0,0], dtype="float32"), k=10)
    assert len(hits) == 2
