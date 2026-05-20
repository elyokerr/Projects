from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_combines_two_ranked_lists_deterministically():
    a = [{"id": "x", "score": 1.0}, {"id": "y", "score": 0.5}]
    b = [{"id": "y", "score": 2.0}, {"id": "z", "score": 1.0}]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [r["id"] for r in fused]
    assert ids[0] == "y"  # appears in both lists
    assert set(ids) == {"x", "y", "z"}


def test_rrf_tiebreak_stable():
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "x"}, {"id": "y"}]
    out = reciprocal_rank_fusion([a, b])
    assert out[0]["id"] == "x" and out[1]["id"] == "y"
