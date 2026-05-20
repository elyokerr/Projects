from src.retrieval.vector_store import ChromaStore


def test_add_and_query_in_memory():
    store = ChromaStore(collection="test", persist_dir=None)
    store.add(
        texts=["the cat sat", "the dog barked"],
        metadatas=[
            {"ticker": "X", "year": 2024, "page": 1},
            {"ticker": "X", "year": 2024, "page": 2},
        ],
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    hits = store.query(query_embedding=[1.0, 0.0], k=1)
    assert hits[0]["metadata"]["page"] == 1
