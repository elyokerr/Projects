from __future__ import annotations
import numpy as np


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in recommended[:k] if it in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(recommended[:k]) if it in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, it in enumerate(recommended[:k]):
        if it in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def coverage(all_reco_lists: list[list[str]], catalogue_size: int) -> float:
    seen = {it for lst in all_reco_lists for it in lst}
    return len(seen) / catalogue_size if catalogue_size else 0.0
