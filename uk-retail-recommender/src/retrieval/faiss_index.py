from __future__ import annotations

import faiss
import numpy as np


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


class FaissIndex:
    """Inner-product FAISS index over L2-normalised vectors (== cosine).

    Verified against faiss 1.14.2: IndexFlatIP.add(vecs) and
    .search(query_2d, k) -> (scores, idxs).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def build(self, vecs: np.ndarray, ids: list[str]) -> "FaissIndex":
        self._index.add(_normalise(vecs))
        self.ids = list(ids)
        return self

    def query(self, vec: np.ndarray, k: int = 50):
        if not self.ids:
            return []
        q = _normalise(vec.reshape(1, -1))
        scores, idxs = self._index.search(q, min(k, len(self.ids)))
        return [(self.ids[int(j)], float(s)) for j, s in zip(idxs[0], scores[0]) if j != -1]
