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


def test_als_no_owned_no_duplicates_no_sentinel_when_k_exceeds_items():
    # User 1 owns all items except "z".  Requesting k=100 (>> n_items) forces
    # implicit 0.7.3 padding.  The fix must strip owned items, duplicates, and
    # sentinel-score padding rows.
    items = [chr(c) for c in range(ord("a"), ord("z"))]  # a..y (25 items)
    rows = []
    for item in items:
        rows.append({"customer_id": 1, "item_id": item, "quantity": 1})
    rows.append({"customer_id": 1, "item_id": "z", "quantity": 1})
    # User 2 buys all items so every item has at least two interactions.
    for item in items + ["z"]:
        rows.append({"customer_id": 2, "item_id": item, "quantity": 1})
    df = pd.DataFrame(rows)
    owned_by_user1 = set(items)  # a..y; "z" is NOT owned
    inter = build_interactions(df)
    model = ALSModel(factors=8, iterations=5, regularization=0.1, random_state=0).fit(inter)
    recs = model.recommend(user_id=1, k=100, filter_owned=True)
    returned_ids = [item_id for item_id, _ in recs]
    returned_scores = [s for _, s in recs]
    # (a) No owned items in results
    assert not any(item_id in owned_by_user1 for item_id in returned_ids), (
        f"Owned items leaked into recs: {[i for i in returned_ids if i in owned_by_user1]}"
    )
    # (b) No duplicate item_ids
    assert len(returned_ids) == len(set(returned_ids)), (
        f"Duplicate item_ids in recs: {returned_ids}"
    )
    # (c) No sentinel scores
    assert all(s > -3.0e38 for s in returned_scores), (
        f"Sentinel score found: {[s for s in returned_scores if s <= -3.0e38]}"
    )
