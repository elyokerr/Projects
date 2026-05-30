from __future__ import annotations

import pandas as pd

_RENAME = {
    "Invoice": "invoice",
    "StockCode": "item_id",
    "Quantity": "quantity",
    "InvoiceDate": "date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_RENAME).copy()
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]
    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int)
    df["item_id"] = df["item_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    cols = ["customer_id", "item_id", "invoice", "date", "quantity", "price", "country"]
    return df[cols].reset_index(drop=True)
