import pandas as pd

from src.retrieval.popularity import PopularityModel


def _df():
    return pd.DataFrame({
        "customer_id":[1,2,3,1], "item_id":["a","a","b","c"],
        "quantity":[1,1,1,1], "invoice":["i"]*4,
        "date":pd.to_datetime(["2010-01-01"]*4), "price":[1.0]*4, "country":["UK"]*4,
    })


def test_returns_top_items_by_purchase_count():
    df = _df()
    m = PopularityModel().fit(df)
    top = m.recommend(k=2)
    assert top[0][0] == "a"  # a bought by 2 customers
    assert len(top) == 2


def test_exclude_filters_items():
    df = _df()
    m = PopularityModel().fit(df)
    top = m.recommend(k=5, exclude={"a"})
    items = [it for it, _ in top]
    assert "a" not in items
    assert set(items) == {"b", "c"}
