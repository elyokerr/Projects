from __future__ import annotations

import pandas as pd


class PopularityModel:
    """Global popularity baseline: rank items by distinct purchasing customers."""

    def __init__(self) -> None:
        self._scores: list[tuple[str, float]] = []

    def fit(self, df: pd.DataFrame) -> PopularityModel:
        counts = df.groupby("item_id")["customer_id"].nunique()
        ranked = counts.sort_values(ascending=False)
        self._scores = [(str(item), float(score)) for item, score in ranked.items()]
        return self

    def recommend(self, k: int = 50, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        for item, score in self._scores:
            if item in exclude:
                continue
            out.append((item, score))
            if len(out) >= k:
                break
        return out
