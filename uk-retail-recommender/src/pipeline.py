from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.interactions import build_interactions
from src.data.split import temporal_split
from src.ranking.features import FEATURE_COLUMNS, build_features
from src.ranking.ranker import Ranker
from src.retrieval.als import ALSModel
from src.retrieval.candidates import SOURCE_NAMES, union_candidates
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.item2item import ItemToItem
from src.retrieval.popularity import PopularityModel

# Path to the optional two-tower item embeddings exported by the Colab run.
# Aligned row-for-row to ``inter.item_ids`` (sorted item order). Loaded when
# present; the pipeline runs without it otherwise.
_TT_EMB_PATH = Path(__file__).resolve().parent.parent / "models" / "two_tower_item_emb.npy"


class RecommenderPipeline:
    """End-to-end retrieval + ranking recommender.

    Built by :meth:`train`, which fits every retrieval generator on the train
    split, learns a LambdaMART ranker using held-out test purchases as labels,
    and stores all per-item / per-customer metadata needed to serve.
    """

    def __init__(
        self,
        *,
        popularity: PopularityModel,
        item2item: ItemToItem,
        als: ALSModel,
        item_meta: pd.DataFrame,
        cust_features: dict[int, dict],
        history: dict[int, set[str]],
        known_customers: set[int],
        ranker: Ranker | None,
        k_per_source: int,
        tt_emb: np.ndarray | None,
        tt_index: FaissIndex | None,
        tt_item_ids: list[str] | None,
    ) -> None:
        self.popularity = popularity
        self.item2item = item2item
        self.als = als
        self.item_meta = item_meta
        self.cust_features = cust_features
        self.history = history
        self.known_customers = known_customers
        self.ranker = ranker
        self.k_per_source = k_per_source
        self.tt_emb = tt_emb
        self.tt_index = tt_index
        self.tt_item_ids = tt_item_ids

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        cutoff: str | pd.Timestamp | None = None,
        *,
        als_factors: int = 32,
        als_iters: int = 8,
        k_per_source: int = 100,
        neg_per_pos: int = 10,  # noqa: ARG003 - reserved for negative sampling tuning
    ) -> RecommenderPipeline:
        if cutoff is None:
            cutoff = df["date"].median()
        cutoff = pd.Timestamp(cutoff)

        train_df, test_df = temporal_split(df, cutoff)

        inter = build_interactions(train_df)

        popularity = PopularityModel().fit(train_df)
        item2item = ItemToItem().fit(train_df)
        als = ALSModel(factors=als_factors, iterations=als_iters).fit(inter)

        # FAISS over ALS item factors (ids aligned to inter.item_ids) for
        # completeness; not used in the default serving path.
        als_factors_mat = als.item_factors
        FaissIndex(als_factors_mat.shape[1]).build(als_factors_mat, inter.item_ids)

        # Optional two-tower embeddings (Colab export). When present they are
        # aligned to inter.item_ids and used as the "tt" retrieval source.
        tt_emb: np.ndarray | None = None
        tt_index: FaissIndex | None = None
        tt_item_ids: list[str] | None = None
        if _TT_EMB_PATH.exists():
            loaded = np.load(_TT_EMB_PATH)
            if loaded.shape[0] == len(inter.item_ids):
                tt_emb = loaded
                tt_item_ids = list(inter.item_ids)
                tt_index = FaissIndex(tt_emb.shape[1]).build(tt_emb, tt_item_ids)

        item_meta = cls._build_item_meta(train_df)
        cust_features = cls._build_cust_features(train_df, cutoff)
        history = (
            train_df.groupby("customer_id")["item_id"]
            .apply(lambda s: {str(it) for it in s})
            .to_dict()
        )
        known_customers = set(history)

        pipe = cls(
            popularity=popularity,
            item2item=item2item,
            als=als,
            item_meta=item_meta,
            cust_features=cust_features,
            history=history,
            known_customers=known_customers,
            ranker=None,
            k_per_source=k_per_source,
            tt_emb=tt_emb,
            tt_index=tt_index,
            tt_item_ids=tt_item_ids,
        )

        pipe.ranker = pipe._fit_ranker(test_df)
        return pipe

    # ------------------------------------------------------------ ranker fit
    def _fit_ranker(self, test_df: pd.DataFrame) -> Ranker | None:
        test_items = (
            test_df.groupby("customer_id")["item_id"]
            .apply(lambda s: {str(it) for it in s})
            .to_dict()
        )

        feature_frames: list[pd.DataFrame] = []
        labels: list[np.ndarray] = []
        groups: list[int] = []

        for customer_id, held_out in test_items.items():
            if customer_id not in self.known_customers:
                continue
            sources = self._candidates_for(customer_id)
            cands = union_candidates(sources, self.k_per_source)
            if not cands:
                continue
            owned = self.history.get(customer_id, set())
            X = build_features(
                cands, self.item_meta, self.cust_features.get(customer_id, {}), owned
            )
            y = X["item_id"].isin(held_out).astype(int).to_numpy()
            # Skip groups with no positives: the held-out item was not
            # retrieved, so the group carries no ranking signal.
            if y.sum() == 0:
                continue
            feature_frames.append(X)
            labels.append(y)
            groups.append(len(X))

        if not feature_frames:
            return None

        X_all = pd.concat(feature_frames, ignore_index=True)
        y_all = np.concatenate(labels)
        return Ranker().fit(X_all[FEATURE_COLUMNS], y_all, group=groups)

    # ----------------------------------------------------------- candidates
    def _candidates_for(self, customer_id: int) -> dict[str, list[tuple[str, float]]]:
        history = list(self.history.get(customer_id, set()))
        sources: dict[str, list[tuple[str, float]]] = {
            "pop": self.popularity.recommend(self.k_per_source, exclude=set(history)),
            "i2i": self.item2item.recommend(history, self.k_per_source, exclude_owned=True),
            "als": (
                self.als.recommend(customer_id, self.k_per_source, filter_owned=True)
                if customer_id in self.als.inter.user_index
                else []
            ),
            "tt": self._tt_source(history, self.k_per_source),
        }
        return sources

    def _tt_source(self, history: list[str], k: int) -> list[tuple[str, float]]:
        if self.tt_index is None or self.tt_emb is None or self.tt_item_ids is None:
            return []
        idx = {it: i for i, it in enumerate(self.tt_item_ids)}
        rows = [idx[it] for it in history if it in idx]
        if not rows:
            return []
        centroid = self.tt_emb[rows].mean(axis=0)
        owned = set(history)
        out: list[tuple[str, float]] = []
        for item_id, score in self.tt_index.query(centroid, k + len(owned)):
            if item_id in owned:
                continue
            out.append((item_id, score))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------- recommend
    def recommend(self, customer_id: int, k: int = 10) -> list[dict]:
        if customer_id not in self.known_customers:
            cold = self.popularity.recommend(k)
            return [
                {"item_id": item_id, "score": float(score), "source": "cold_start"}
                for item_id, score in cold
            ]

        sources = self._candidates_for(customer_id)
        cands = union_candidates(sources, self.k_per_source)
        owned = self.history.get(customer_id, set())

        if not cands or self.ranker is None:
            # Fall back to popularity when no candidates / no trained ranker.
            cold = self.popularity.recommend(k, exclude=owned)
            return [
                {"item_id": item_id, "score": float(score), "source": "pop"}
                for item_id, score in cold
            ]

        X = build_features(
            cands, self.item_meta, self.cust_features.get(customer_id, {}), owned
        )
        scores = self.ranker.predict(X[FEATURE_COLUMNS])

        # Per-source max over the candidate pool, so attribution compares each
        # source on its own scale (popularity counts dwarf ALS/cosine scores
        # otherwise, and every item would be tagged "pop").
        source_max = {
            name: max((c.get(name, 0.0) for c in cands), default=0.0)
            for name in SOURCE_NAMES
        }

        def _dominant_source(cand: dict) -> str:
            best_name, best_norm = "pop", -1.0
            for name in SOURCE_NAMES:
                raw = cand.get(name, 0.0)
                if raw <= 0.0:
                    continue
                norm = raw / source_max[name] if source_max[name] > 0 else 0.0
                if norm > best_norm:
                    best_name, best_norm = name, norm
            return best_name

        ranked: list[dict] = []
        for cand, rank_score in zip(cands, scores, strict=True):
            item_id = cand["item_id"]
            if item_id in owned:  # double-check: never recommend an owned item
                continue
            dominant = _dominant_source(cand)
            ranked.append(
                {
                    "item_id": item_id,
                    "score": float(rank_score),
                    "source": dominant,
                }
            )

        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked[:k]

    # -------------------------------------------------------------- helpers
    def known_customer(self) -> int:
        return next(iter(self.known_customers))

    @staticmethod
    def _build_item_meta(train_df: pd.DataFrame) -> pd.DataFrame:
        grouped = train_df.groupby("item_id")
        meta = pd.DataFrame(
            {
                "price": grouped["price"].mean(),
                "popularity": grouped["customer_id"].nunique(),
            }
        ).reset_index()
        meta["item_id"] = meta["item_id"].astype(str)
        return meta

    @staticmethod
    def _build_cust_features(train_df: pd.DataFrame, cutoff: pd.Timestamp) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for customer_id, g in train_df.groupby("customer_id"):
            last_purchase = g["date"].max()
            recency_days = float((cutoff - last_purchase).days)
            frequency = int(g["invoice"].nunique())
            monetary = float((g["quantity"] * g["price"]).sum())
            country = g["country"].mode()
            country = str(country.iloc[0]) if len(country) else ""
            out[int(customer_id)] = {
                "recency_days": recency_days,
                "frequency": frequency,
                "monetary": monetary,
                "country": country,
            }
        return out
