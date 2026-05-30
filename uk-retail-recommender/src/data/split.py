from __future__ import annotations

import pandas as pd


def temporal_split(df: pd.DataFrame, cutoff: str | pd.Timestamp):
    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    seen = set(train["customer_id"])
    test = test[test["customer_id"].isin(seen)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
