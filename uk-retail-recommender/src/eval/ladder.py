"""End-to-end ladder evaluation: compare retrieval rungs on a single DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.interactions import build_interactions
from src.data.split import temporal_split
from src.eval.evaluate import evaluate_recommender, retrieval_ceiling
from src.pipeline import RecommenderPipeline
from src.retrieval.als import ALSModel
from src.retrieval.candidates import union_candidates
from src.retrieval.item2item import ItemToItem
from src.retrieval.popularity import PopularityModel

_TT_EMB_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "two_tower_item_emb.npy"

# ALS is trained with modest settings so the ladder runs in a few seconds on
# the small sample fixture.
_ALS_FACTORS = 32
_ALS_ITERS = 8


def evaluate_ladder(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp | None = None,
    k: int = 10,
) -> dict:
    """Evaluate each retrieval rung and the two-stage pipeline on df.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame (output of clean_transactions).
    cutoff:
        Temporal split boundary.  None -> median date.
    k:
        Recommendation cutoff used for all metrics.

    Returns
    -------
    dict with keys:
        <rung_name>: {"recall@k", "ndcg@k", "map@k", "coverage"}
        "retrieval_ceiling": float
        "k": int
        "n_users": int
    """
    if cutoff is None:
        cutoff = df["date"].median()
    cutoff = pd.Timestamp(cutoff)

    train_df, test_df = temporal_split(df, cutoff)

    # Build truth: customers present in train who also appear in test
    truth: dict[int, set[str]] = (
        test_df.groupby("customer_id")["item_id"]
        .apply(lambda s: {str(it) for it in s})
        .to_dict()
    )
    # Keep only customers whose test set is non-empty
    truth = {uid: items for uid, items in truth.items() if items}

    # Fit shared models once so all rungs use the same trained objects
    popularity = PopularityModel().fit(train_df)
    item2item = ItemToItem().fit(train_df)
    inter = build_interactions(train_df)
    als = ALSModel(factors=_ALS_FACTORS, iterations=_ALS_ITERS).fit(inter)

    # Per-customer purchase history from train
    history: dict[int, set[str]] = (
        train_df.groupby("customer_id")["item_id"]
        .apply(lambda s: {str(it) for it in s})
        .to_dict()
    )

    # Popularity rung: global top-k excluding owned items
    def pop_reco(uid: int, n: int) -> list[str]:
        owned = history.get(uid, set())
        return [item_id for item_id, _ in popularity.recommend(n, exclude=owned)]

    # Item-to-item rung
    def i2i_reco(uid: int, n: int) -> list[str]:
        hist = list(history.get(uid, set()))
        return [item_id for item_id, _ in item2item.recommend(hist, n, exclude_owned=True)]

    # ALS rung
    def als_reco(uid: int, n: int) -> list[str]:
        if uid not in als.inter.user_index:
            return []
        return [item_id for item_id, _ in als.recommend(uid, n, filter_owned=True)]

    # Two-stage pipeline: pass the full df with the same cutoff so the pipeline
    # performs its own identical split and has access to test-period rows for
    # LambdaMART training.  Passing train_df would leave the internal test
    # split empty (all rows <= cutoff) and prevent the ranker from training.
    pipeline = RecommenderPipeline.train(df, cutoff=cutoff)

    def two_stage_reco(uid: int, n: int) -> list[str]:
        return [rec["item_id"] for rec in pipeline.recommend(uid, n)]

    rungs: dict[str, object] = {
        "popularity": pop_reco,
        "item2item": i2i_reco,
        "als": als_reco,
        "two_stage": two_stage_reco,
    }

    # Optional two-tower rung when pre-trained embeddings are present
    import numpy as np  # noqa: PLC0415 - lazy import to avoid hard dep at module level

    if _TT_EMB_PATH.exists():
        from src.retrieval.faiss_index import FaissIndex  # noqa: PLC0415

        tt_emb = np.load(_TT_EMB_PATH)
        if tt_emb.shape[0] == len(inter.item_ids):
            tt_index = FaissIndex(tt_emb.shape[1]).build(tt_emb, list(inter.item_ids))
            tt_item_map = {it: i for i, it in enumerate(inter.item_ids)}

            def tt_reco(uid: int, n: int) -> list[str]:
                hist = [it for it in history.get(uid, set()) if it in tt_item_map]
                if not hist:
                    return []
                owned = history.get(uid, set())
                centroid = tt_emb[[tt_item_map[it] for it in hist]].mean(axis=0)
                out: list[str] = []
                for item_id, _ in tt_index.query(centroid, n + len(owned)):
                    if item_id not in owned:
                        out.append(item_id)
                    if len(out) >= n:
                        break
                return out

            rungs["two_tower"] = tt_reco

    results: dict = {}
    for name, reco_fn in rungs.items():
        res = evaluate_recommender(reco_fn, truth, k)
        results[name] = {
            "recall@k": res["recall@k"],
            "ndcg@k": res["ndcg@k"],
            "map@k": res["map@k"],
            "coverage": res["coverage"],
        }

    # Retrieval ceiling using the union of all sources from the pipeline
    k_per_source = pipeline.k_per_source

    def candidate_fn(uid: int) -> list[str]:
        sources = {
            "pop": popularity.recommend(k_per_source, exclude=history.get(uid, set())),
            "i2i": item2item.recommend(
                list(history.get(uid, set())), k_per_source, exclude_owned=True
            ),
            "als": (
                als.recommend(uid, k_per_source, filter_owned=True)
                if uid in als.inter.user_index
                else []
            ),
            "tt": [],
        }
        cands = union_candidates(sources, k_per_source)
        return [c["item_id"] for c in cands]

    results["retrieval_ceiling"] = retrieval_ceiling(candidate_fn, truth)
    results["k"] = k
    results["n_users"] = len(truth)

    return results
