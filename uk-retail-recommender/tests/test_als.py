import pandas as pd

from src.data.interactions import build_interactions
from src.retrieval.als import ALSModel


def test_als_fits_and_recommends():
    # two clusters: users 1,2 buy a,b ; user 3 buys c
    df = pd.DataFrame({
        "customer_id":[1,1,2,2,3,3], "item_id":["a","b","a","b","c","c"],
        "quantity":[1,1,1,1,1,1],
    })
    inter = build_interactions(df)
    model = ALSModel(factors=8, iterations=5, regularization=0.1, random_state=0).fit(inter)
    assert model.user_factors.shape == (3, 8)
    assert model.item_factors.shape == (3, 8)
    recs = model.recommend(user_id=1, k=3)
    assert isinstance(recs, list) and len(recs) >= 1
    assert all(isinstance(it, str) for it, _ in recs)
