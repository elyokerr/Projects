from src.pipeline import RecommenderPipeline


def test_pipeline_recommends_and_handles_unknown(sample_df):
    # Small als_iters keeps the test fast; cutoff=None -> auto median date.
    pipe = RecommenderPipeline.train(sample_df, cutoff=None, als_iters=3)

    recs = pipe.recommend(pipe.known_customer(), k=5)
    assert 0 < len(recs) <= 5
    owned = pipe.history[pipe.known_customer()]
    for rec in recs:
        assert "item_id" in rec
        assert "score" in rec
        assert isinstance(rec["score"], float)
        assert rec["item_id"] not in owned  # never recommend an owned item

    # Unknown customer -> non-empty popularity cold-start.
    cold = pipe.recommend(-999, k=5)
    assert len(cold) > 0
    assert all(r["source"] == "cold_start" for r in cold)


def test_pipeline_ranker_trained_on_sample(sample_df):
    # The sample has enough retrieved positives to train the ranker.
    pipe = RecommenderPipeline.train(sample_df, cutoff=None, als_iters=3)
    assert pipe.ranker is not None
