import numpy as np

from src.eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, k=3) == 0.5  # 1 of 2 relevant found


def test_ndcg_at_k_perfect_and_partial():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0
    # relevant only at position 2: DCG=1/log2(3), IDCG=1 -> ~0.6309
    val = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert abs(val - (1 / np.log2(3))) < 1e-9


def test_average_precision():
    # hits at ranks 1 and 3 -> (1/1 + 2/3)/2
    ap = average_precision_at_k(["a", "x", "b"], {"a", "b"}, k=3)
    assert abs(ap - ((1.0 + 2 / 3) / 2)) < 1e-9
