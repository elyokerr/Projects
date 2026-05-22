"""BM25 keyword index using the `bm25s` library, with disk persistence."""

from pathlib import Path
from typing import List
import pickle

import bm25s


class BM25Index:
    """In-process BM25 index over a static document corpus."""

    def __init__(self):
        self.retriever: bm25s.BM25 | None = None
        self.ids: List[str] = []

    def build(self, docs: List[dict]) -> None:
        """`docs` is a list of {'id': str, 'text': str}."""
        self.ids = [d["id"] for d in docs]
        corpus_tokens = bm25s.tokenize(
            [d["text"] for d in docs],
            stopwords="en",
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def query(self, q: str, k: int = 50) -> List[dict]:
        q_tokens = bm25s.tokenize([q], stopwords="en")
        idxs, scores = self.retriever.retrieve(q_tokens, k=k)
        return [
            {"id": self.ids[i], "score": float(s)}
            for i, s in zip(idxs[0], scores[0])
        ]

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps({"retriever": self.retriever, "ids": self.ids}))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = pickle.loads(path.read_bytes())
        self = cls()
        self.retriever = data["retriever"]
        self.ids = data["ids"]
        return self
