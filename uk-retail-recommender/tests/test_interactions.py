import pandas as pd
from src.data.interactions import build_interactions


def test_builds_csr_with_mappings():
    df = pd.DataFrame({"customer_id": [1, 1, 2], "item_id": ["a", "b", "a"], "quantity": [2, 1, 5]})
    inter = build_interactions(df)
    assert inter.matrix.shape == (2, 2)
    ai = inter.item_index["a"]
    u1 = inter.user_index[1]
    assert inter.matrix[u1, ai] == 2
    assert inter.matrix[inter.user_index[2], ai] == 5
