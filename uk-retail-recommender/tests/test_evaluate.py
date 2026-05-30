from src.eval.evaluate import evaluate_recommender, retrieval_ceiling


def test_evaluate_aggregates_over_users():
    truth = {1: {"a", "b"}, 2: {"c"}}

    def reco(uid, k):
        return {1: ["a", "x"], 2: ["c", "y"]}[uid][:k]

    res = evaluate_recommender(reco, truth, k=2)
    assert 0.0 <= res["recall@k"] <= 1.0
    assert res["n_users"] == 2


def test_retrieval_ceiling():
    truth = {1: {"a", "b"}}

    def cand(uid):
        return ["a", "z"]  # only 'a' retrievable

    assert retrieval_ceiling(cand, truth) == 0.5
