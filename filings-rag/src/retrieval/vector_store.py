"""ChromaDB vector store wrapper with cosine similarity."""

from pathlib import Path
from typing import List, Optional

import chromadb


class ChromaStore:
    """Persistent (or in-memory) Chroma collection with cosine similarity."""

    def __init__(self, collection: str, persist_dir: Optional[Path]):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self.client = chromadb.EphemeralClient()
        self.coll = self.client.get_or_create_collection(
            collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts, metadatas, ids, embeddings):
        self.coll.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def query(self, query_embedding, k: int = 10) -> List[dict]:
        res = self.coll.query(query_embeddings=[query_embedding], n_results=k)
        return [
            {"id": _id, "text": doc, "metadata": meta, "score": 1 - dist}
            for _id, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]
