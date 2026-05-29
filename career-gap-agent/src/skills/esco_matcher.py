from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], list[np.ndarray]]


def _default_embed_fn() -> EmbedFn:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(xs: Sequence[str]) -> list[np.ndarray]:
        out = []
        for v in model.embed(list(xs)):
            arr = np.asarray(v, dtype=np.float32)
            out.append(arr / (np.linalg.norm(arr) + 1e-9))
        return out

    return embed


class EscoMatcher:
    """Map free-text skill phrases to canonical ESCO skill labels by cosine similarity."""

    def __init__(self, index, embed_fn: EmbedFn | None = None):
        self.index = index
        self.embed_fn = embed_fn or _default_embed_fn()

    def match(self, phrases: Sequence[str], threshold: float = 0.62) -> list[str]:
        if not phrases:
            return []
        vecs = self.embed_fn(phrases)
        seen: dict[str, None] = {}
        for v in vecs:
            sims = self.index.embeddings @ v  # index rows are unit-normalised
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                seen.setdefault(self.index.labels[best], None)
        return list(seen.keys())
