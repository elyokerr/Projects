import numpy as np
import pandas as pd

from src.data.interactions import build_interactions
from src.retrieval.two_tower import TwoTowerModel


def _inter():
    df = pd.DataFrame({
        "customer_id":[1,1,2,2,3], "item_id":["a","b","a","b","c"], "quantity":[1]*5,
    })
    return build_interactions(df)


def test_two_tower_trains_and_embeds():
    inter = _inter()
    m = TwoTowerModel(dim=8, epochs=3, seed=0).fit(inter)
    item_emb = m.item_embeddings()
    assert item_emb.shape == (len(inter.item_ids), 8)
    assert np.isfinite(item_emb).all()
    user_emb = m.user_embeddings()
    assert user_emb.shape == (len(inter.user_ids), 8)
    assert np.isfinite(user_emb).all()


def test_two_tower_is_deterministic():
    inter = _inter()
    a = TwoTowerModel(dim=8, epochs=3, seed=0).fit(inter).item_embeddings()
    b = TwoTowerModel(dim=8, epochs=3, seed=0).fit(inter).item_embeddings()
    assert np.allclose(a, b)


def test_two_tower_loss_decreases():
    inter = _inter()
    m = TwoTowerModel(dim=8, epochs=15, seed=0).fit(inter)
    assert m.loss_history_[-1] < m.loss_history_[0]
