"""BGE embeddings wrapper.

Uses `fastembed` (ONNX-optimised) so embedding runs ~80x faster than vanilla
sentence-transformers on the same model — fast enough on CPU that no GPU /
Colab roundtrip is needed. Same model (`BAAI/bge-small-en-v1.5`), same
384-dim L2-normalised output space.
"""

from typing import List

from fastembed import TextEmbedding


class BGEEmbedder:
    """Wraps `fastembed.TextEmbedding` behind a tiny stable interface."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",  # kept for back-compat; fastembed picks CPU/GPU automatically
    ):
        self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.embed(texts)]
