from src.retrieval.bm25_index import BM25Index


def test_bm25_recovers_keyword_doc(tmp_path):
    docs = [
        {"id": "a", "text": "going concern statement principal risks"},
        {"id": "b", "text": "the cat sat on the mat"},
    ]
    idx = BM25Index()
    idx.build(docs)
    hits = idx.query("going concern", k=1)
    assert hits[0]["id"] == "a"
    idx.save(tmp_path / "bm25.pkl")
    fresh = BM25Index.load(tmp_path / "bm25.pkl")
    assert fresh.query("going concern", k=1)[0]["id"] == "a"
