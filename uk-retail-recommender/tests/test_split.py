import pandas as pd
from src.data.split import temporal_split


def _txn():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01", "2010-04-01"])
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 2],
        "item_id": ["a", "b", "c", "d"],
        "invoice": ["i1", "i2", "i3", "i4"],
        "date": dates,
        "quantity": [1, 1, 1, 1],
        "price": [1.0, 1, 1, 1],
        "country": ["UK"] * 4,
    })


def test_split_is_temporal_and_leak_free():
    train, test = temporal_split(_txn(), cutoff="2010-02-15")
    assert train["date"].max() <= pd.Timestamp("2010-02-15")
    assert test["date"].min() > pd.Timestamp("2010-02-15")
    # leakage guard: no test row at or before cutoff
    assert (test["date"] > pd.Timestamp("2010-02-15")).all()


def test_test_only_keeps_customers_seen_in_train():
    # customer 2 only appears after cutoff -> excluded from test (no train history)
    train, test = temporal_split(_txn(), cutoff="2010-02-15")
    assert set(test["customer_id"]).issubset(set(train["customer_id"]))
