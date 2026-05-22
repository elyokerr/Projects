"""BGE cross-encoder re-ranker for high-precision top-k re-ranking.

Uses sentence-transformers' CrossEncoder (rather than FlagEmbedding) because
FlagEmbedding has a compatibility bug with transformers>=5 — it relies on
`tokenizer.prepare_for_model` which has been removed. CrossEncoder loads the
exact same BGE re-ranker model from HuggingFace and is maintained.
"""

from typing import List

from sentence_transformers import CrossEncoder


class BGEReranker:
    """Cross-encoder reranker. Same `bge-reranker-v2-m3` model, different loader."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """Return top_k candidates sorted by cross-encoder relevance score."""
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        return sorted(scored, key=lambda x: -x["rerank_score"])[:top_k]
