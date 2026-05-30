from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


class ItemToItem:
    """Co-purchase item-to-item similarity.

    Builds a symmetric co-occurrence count over items appearing in the same
    invoice, then normalises with a cosine-style weight:
        sim(i, j) = count(i, j) / sqrt(count(i) * count(j))
    where count(i) is the number of baskets containing item i.
    """

    def __init__(self) -> None:
        self._neighbours: dict[str, list[tuple[str, float]]] = {}

    def fit(self, df: pd.DataFrame) -> "ItemToItem":
        item_freq: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for _, basket in df.groupby("invoice"):
            items = sorted({str(it) for it in basket["item_id"]})
            for it in items:
                item_freq[it] += 1
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1

        sims: dict[str, dict[str, float]] = defaultdict(dict)
        for (a, b), c in pair_counts.items():
            denom = math.sqrt(item_freq[a] * item_freq[b])
            if denom <= 0:
                continue
            s = c / denom
            sims[a][b] = s
            sims[b][a] = s

        self._neighbours = {
            item: sorted(neigh.items(), key=lambda kv: kv[1], reverse=True)
            for item, neigh in sims.items()
        }
        return self

    def similar(self, item: str, k: int = 50) -> list[tuple[str, float]]:
        return self._neighbours.get(str(item), [])[:k]

    def recommend(
        self,
        history: list[str],
        k: int = 50,
        exclude_owned: bool = True,
    ) -> list[tuple[str, float]]:
        owned = {str(it) for it in history}
        scores: dict[str, float] = defaultdict(float)
        for item in owned:
            for neigh, s in self._neighbours.get(item, []):
                scores[neigh] += s
        if exclude_owned:
            scores = {it: s for it, s in scores.items() if it not in owned}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]
