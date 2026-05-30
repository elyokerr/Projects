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

    def fit(self, inter: Interactions) -> "ALSModel":
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
        ids, scores = self._m.recommend(
            uidx,
            self.inter.matrix[uidx],
            N=k,
            filter_already_liked_items=filter_owned,
        )
        return [(self.inter.item_ids[int(i)], float(s)) for i, s in zip(ids, scores)]
