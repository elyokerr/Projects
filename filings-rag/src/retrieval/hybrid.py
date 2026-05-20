"""Reciprocal Rank Fusion for combining dense + BM25 result lists."""

from collections import defaultdict
from typing import List


def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
) -> List[dict]:
    """Fuse multiple ranked lists of `{'id': ...}` dicts via RRF.

    Ties break by first-seen rank for stability.
    """
    scores: dict = defaultdict(float)
    first_rank: dict = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            _id = item["id"]
            scores[_id] += 1.0 / (k + rank)
            first_rank.setdefault(_id, rank)
    return [
        {"id": _id, "score": s}
        for _id, s in sorted(
            scores.items(),
            key=lambda kv: (-kv[1], first_rank[kv[0]]),
        )
    ]
