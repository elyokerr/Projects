"""BGE cross-encoder re-ranker for high-precision top-k re-ranking."""

from typing import List

from FlagEmbedding import FlagReranker


class BGEReranker:
    """Wrap `FlagReranker` so the calling code stays library-agnostic."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = FlagReranker(model_name, use_fp16=False)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """Return top_k candidates sorted by cross-encoder relevance score."""
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        scored = [
            {**c, "rerank_score": s}
            for c, s in zip(candidates, scores)
        ]
        return sorted(scored, key=lambda x: -x["rerank_score"])[:top_k]
