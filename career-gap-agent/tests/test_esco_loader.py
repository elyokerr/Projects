import numpy as np

from src.data.esco_loader import EscoIndex


def test_loads_committed_index_and_dims_align():
    idx = EscoIndex.load()
    assert len(idx.labels) >= 50  # stub index (override 1)
    assert idx.embeddings.shape[0] == len(idx.labels)
    assert idx.embeddings.shape[1] == 384  # bge-small dim


def test_embeddings_are_unit_normalised():
    idx = EscoIndex.load()
    norms = np.linalg.norm(idx.embeddings[:50], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
