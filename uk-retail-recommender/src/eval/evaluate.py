from __future__ import annotations

from collections.abc import Callable

from src.eval.metrics import average_precision_at_k, coverage, ndcg_at_k, recall_at_k


def evaluate_recommender(
    reco_fn: Callable[[int, int], list[str]],
    truth: dict[int, set[str]],
    k: int,
) -> dict[str, float]:
    """Average recall/ndcg/map over users and compute catalogue coverage.

    Args:
        reco_fn: callable(user_id, k) -> list[str] of recommended item ids.
        truth: dict mapping user_id to set of relevant item ids.
        k: cutoff for all metrics.

    Returns:
        dict with recall@k, ndcg@k, map@k, coverage, n_users.
    """
    recalls, ndcgs, aps, all_recs = [], [], [], []
    for uid, relevant in truth.items():
        recs = reco_fn(uid, k)
        all_recs.append(recs)
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        aps.append(average_precision_at_k(recs, relevant, k))

    n = len(truth)
    # catalogue size = all unique items that appear as relevant
    catalogue = {it for relevant in truth.values() for it in relevant}

    return {
        "recall@k": sum(recalls) / n if n else 0.0,
        "ndcg@k": sum(ndcgs) / n if n else 0.0,
        "map@k": sum(aps) / n if n else 0.0,
        "coverage": coverage(all_recs, len(catalogue)),
        "n_users": n,
    }


def retrieval_ceiling(
    candidate_fn: Callable[[int], list[str]],
    truth: dict[int, set[str]],
) -> float:
    """Mean over users of |candidates ∩ relevant| / |relevant|.

    Args:
        candidate_fn: callable(user_id) -> list[str] of candidate item ids.
        truth: dict mapping user_id to set of relevant item ids.

    Returns:
        Mean recall of the candidate set (upper bound for any ranker).
    """
    scores = []
    for uid, relevant in truth.items():
        if not relevant:
            continue
        candidates = set(candidate_fn(uid))
        scores.append(len(candidates & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0
