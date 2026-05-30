from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Interactions:
    """Customer-by-item implicit matrix plus id<->index mappings."""

    matrix: sp.csr_matrix  # shape (n_users, n_items), values = purchase counts
    user_ids: list[int]  # row index -> customer_id
    item_ids: list[str]  # col index -> stock_code
    user_index: dict[int, int]  # customer_id -> row index
    item_index: dict[str, int]  # stock_code -> col index


def build_interactions(df: pd.DataFrame) -> Interactions:
    user_ids = sorted(df["customer_id"].unique().tolist())
    item_ids = sorted(df["item_id"].unique().tolist())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {it: j for j, it in enumerate(item_ids)}
    rows = df["customer_id"].map(user_index).to_numpy()
    cols = df["item_id"].map(item_index).to_numpy()
    vals = df["quantity"].to_numpy(dtype="float32")
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids))).tocsr()
    mat.sum_duplicates()
    return Interactions(mat, user_ids, item_ids, user_index, item_index)
