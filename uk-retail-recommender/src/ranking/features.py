from __future__ import annotations

import pandas as pd

# Single source of truth for feature order.  All downstream code
# (ranker, pipeline) must select/build columns in this order.
FEATURE_COLUMNS: list[str] = [
    "pop",
    "als",
    "i2i",
    "tt",
    "price",
    "popularity",
    "recency_days",
    "frequency",
    "monetary",
    "prev_bought",
    "country_code",
]

# Country name -> integer code mapping.  Built lazily from training data or
# supplied at call time.  Unknown countries map to -1.
_COUNTRY_MAP: dict[str, int] = {
    "United Kingdom": 0,
    "Germany": 1,
    "France": 2,
    "EIRE": 3,
    "Spain": 4,
    "Netherlands": 5,
    "Belgium": 6,
    "Switzerland": 7,
    "Portugal": 8,
    "Australia": 9,
    "Norway": 10,
    "Italy": 11,
    "Channel Islands": 12,
    "Finland": 13,
    "Cyprus": 14,
    "Sweden": 15,
    "Austria": 16,
    "Denmark": 17,
    "Japan": 18,
    "Poland": 19,
    "Israel": 20,
    "USA": 21,
    "Hong Kong": 22,
    "Singapore": 23,
    "Iceland": 24,
    "Canada": 25,
    "Greece": 26,
    "Malta": 27,
    "United Arab Emirates": 28,
    "Lebanon": 29,
    "Lithuania": 30,
    "Brazil": 31,
    "Czech Republic": 32,
    "Bahrain": 33,
    "Saudi Arabia": 34,
    "UK": 35,
}


def build_features(
    candidates: list[dict],
    item_meta: pd.DataFrame,
    cust_features: dict,
    owned_items: set[str],
) -> pd.DataFrame:
    """Build a feature matrix with one row per candidate.

    Parameters
    ----------
    candidates:
        List of dicts from ``union_candidates``.  Each has ``item_id`` plus
        per-source score keys ``pop``, ``als``, ``i2i``, ``tt``.
    item_meta:
        DataFrame with at least columns ``item_id``, ``price``, ``popularity``.
    cust_features:
        Dict with keys ``recency_days``, ``frequency``, ``monetary``,
        ``country`` (str).
    owned_items:
        Set of item_ids the customer has already purchased.

    Returns
    -------
    pd.DataFrame with columns ``item_id`` + ``FEATURE_COLUMNS``, one row per
    candidate.  ``item_id`` is NOT in ``FEATURE_COLUMNS`` and is kept for
    downstream joining only.
    """
    df = pd.DataFrame(candidates)

    # Merge item metadata
    meta = item_meta[["item_id", "price", "popularity"]].copy()
    df = df.merge(meta, on="item_id", how="left")
    df["price"] = df["price"].fillna(0.0)
    df["popularity"] = df["popularity"].fillna(0)

    # Broadcast customer-level features
    df["recency_days"] = float(cust_features.get("recency_days", 0))
    df["frequency"] = float(cust_features.get("frequency", 0))
    df["monetary"] = float(cust_features.get("monetary", 0.0))

    # prev_bought flag
    df["prev_bought"] = df["item_id"].isin(owned_items).astype(int)

    # Country code
    country_str = cust_features.get("country", "")
    df["country_code"] = int(_COUNTRY_MAP.get(country_str, -1))

    # Ensure all source columns exist and are filled
    for col in ("pop", "als", "i2i", "tt"):
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df[["item_id"] + FEATURE_COLUMNS]
