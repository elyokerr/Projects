"""BGE embeddings wrapper around sentence-transformers."""

from typing import List

from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """Thin wrapper that normalises embeddings for cosine similarity."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
    ):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
