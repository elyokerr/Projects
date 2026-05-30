from __future__ import annotations

import os

# Quiet OpenBLAS threadpool warning and avoid the perf pitfall flagged by implicit.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from implicit.als import AlternatingLeastSquares  # noqa: E402

from src.data.interactions import Interactions  # noqa: E402


class ALSModel:
    """Thin wrapper over implicit's AlternatingLeastSquares.

    Verified against implicit 0.7.3:
      - .fit(user_items)         user_items CSR has users as rows
      - .recommend(userid, user_items, N=..., filter_already_liked_items=...)
    """

    def __init__(self, factors=64, iterations=15, regularization=0.05, random_state=0):
        self._m = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
        )
        self.inter: Interactions | None = None

    def fit(self, inter: Interactions) -> ALSModel:
        # implicit >=0.5 expects user_items (users as rows) for .fit
        self._m.fit(inter.matrix, show_progress=False)
        self.inter = inter
        return self

    @property
    def user_factors(self) -> np.ndarray:
        return np.asarray(self._m.user_factors)

    @property
    def item_factors(self) -> np.ndarray:
        return np.asarray(self._m.item_factors)

    def recommend(self, user_id: int, k: int = 50, filter_owned: bool = True):
        assert self.inter is not None, "call fit() before recommend()"
        uidx = self.inter.user_index[user_id]
        # Cap N at the number of items so implicit 0.7.3 does not pad results
        # with sentinel scores (-FLT_MAX) and repeated indices.
        n = min(k, self.inter.matrix.shape[1])
        ids, scores = self._m.recommend(
            uidx,
            self.inter.matrix[uidx],
            N=n,
            filter_already_liked_items=filter_owned,
        )
        _SENTINEL = -3.0e38
        seen: set[str] = set()
        result: list[tuple[str, float]] = []
        for i, s in zip(ids, scores, strict=False):
            # Drop sentinel padding rows injected when N > available items.
            if float(s) <= _SENTINEL:
                continue
            item_id = self.inter.item_ids[int(i)]
            # Dedup by item_id, keeping first (highest-ranked) occurrence.
            if item_id in seen:
                continue
            seen.add(item_id)
            result.append((item_id, float(s)))
        return result
