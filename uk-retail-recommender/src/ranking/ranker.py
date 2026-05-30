from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ranking.features import FEATURE_COLUMNS  # single source of truth

__all__ = ["Ranker", "FEATURE_COLUMNS"]


class Ranker:
    """LightGBM LambdaMART ranker wrapping LGBMRanker.

    Usage
    -----
    r = Ranker().fit(X[FEATURE_COLUMNS], y, groups)
    scores = r.predict(X[FEATURE_COLUMNS])
    """

    def __init__(self) -> None:
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=0,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group: list[int],
    ) -> Ranker:
        """Fit the ranker.

        Parameters
        ----------
        X:
            Feature matrix; columns must match FEATURE_COLUMNS.
        y:
            Integer relevance labels (0/1 or graded); length == len(X).
        group:
            List of per-query group sizes summing to len(X).
        """
        self._model.fit(X[FEATURE_COLUMNS], y, group=group)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw ranking scores (higher = more relevant)."""
        return self._model.predict(X[FEATURE_COLUMNS])
