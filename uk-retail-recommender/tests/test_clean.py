import pandas as pd
from src.data.clean import clean_transactions


def _raw():
    return pd.DataFrame({
        "Invoice": ["536365", "C536379", "536366", "536367"],
        "StockCode": ["85123A", "22423", "84406B", "21730"],
        "Description": ["a", "b", "c", "d"],
        "Quantity": [6, -1, 0, 3],
        "InvoiceDate": pd.to_datetime(["2010-12-01", "2010-12-01", "2010-12-02", "2010-12-02"]),
        "Price": [2.55, 1.0, 1.0, 4.25],
        "Customer ID": [17850, 17850, None, 17851],
        "Country": ["UK", "UK", "UK", "UK"],
    })


def test_drops_cancellation_nonpositive_and_null_customer():
    out = clean_transactions(_raw())
    # row0 keeps (qty 6, customer present); C-invoice, qty<=0, and null customer all dropped
    assert len(out) == 2
    assert set(out["item_id"]) == {"85123A", "21730"}
    assert out["customer_id"].dtype.kind in ("i", "u")
    assert list(out.columns) >= ["customer_id", "item_id", "invoice", "date", "quantity", "price", "country"]
