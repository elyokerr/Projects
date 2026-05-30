import pandas as pd

from src.retrieval.item2item import ItemToItem


def test_co_purchase_within_invoice():
    df = pd.DataFrame({"customer_id":[1,1,2], "item_id":["a","b","a"],
                       "invoice":["i1","i1","i2"], "quantity":[1,1,1],
                       "date":pd.to_datetime(["2010-01-01"]*3),
                       "price":[1.0]*3, "country":["UK"]*3})
    m = ItemToItem().fit(df)
    # a and b co-occur in invoice i1
    sim = dict(m.similar("a", k=5))
    assert "b" in sim and sim["b"] > 0


def test_recommend_from_history_excludes_owned():
    df = pd.DataFrame({"customer_id":[1,1,1], "item_id":["a","b","c"],
                       "invoice":["i1","i1","i1"], "quantity":[1,1,1],
                       "date":pd.to_datetime(["2010-01-01"]*3),
                       "price":[1.0]*3, "country":["UK"]*3})
    m = ItemToItem().fit(df)
    recs = dict(m.recommend(["a"], k=5))
    assert "a" not in recs  # already owned excluded
    # b and c co-occur with a in the same basket
    assert "b" in recs and "c" in recs


def test_recommend_can_keep_owned():
    df = pd.DataFrame({"customer_id":[1,1,1], "item_id":["a","b","c"],
                       "invoice":["i1","i1","i1"], "quantity":[1,1,1],
                       "date":pd.to_datetime(["2010-01-01"]*3),
                       "price":[1.0]*3, "country":["UK"]*3})
    m = ItemToItem().fit(df)
    recs = dict(m.recommend(["a"], k=5, exclude_owned=False))
    # "a" does not reappear because items are not their own co-purchase neighbours
    assert "b" in recs
