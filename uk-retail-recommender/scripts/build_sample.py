"""Build the committed test/demo sample. Uses data/raw parquet if present, else synthesises."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so src imports work when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.clean import clean_transactions  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_transactions.parquet"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "online_retail_II.parquet"


def _synth(n_customers=120, n_items=80, seed=0, n_clusters=6, in_cluster_prob=0.85):
    """Synthesise transactions with latent cluster structure.

    Each customer belongs to one of n_clusters latent clusters.  Items are
    similarly partitioned.  With probability in_cluster_prob each line item
    drawn for a basket comes from the customer's own cluster; otherwise it is
    drawn uniformly from all items.

    Each order is a basket with 1-4 line items (same invoice number) so that
    item2item co-occurrence has real signal to exploit.  This gives ALS and
    item2item the collaborative structure needed to meaningfully outperform
    the global popularity baseline.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-01-01")

    # Assign each customer to a cluster
    cust_cluster = rng.integers(0, n_clusters, size=n_customers)

    # Partition item catalogue into clusters (round-robin so sizes are even)
    item_clusters: list[list[str]] = [[] for _ in range(n_clusters)]
    for i in range(n_items):
        item_clusters[i % n_clusters].append(f"SKU{i:03d}")
    all_items = [f"SKU{i:03d}" for i in range(n_items)]

    def _draw_item(cluster: int) -> str:
        if rng.random() < in_cluster_prob:
            ci = item_clusters[cluster]
            return ci[int(rng.integers(0, len(ci)))]
        return all_items[int(rng.integers(0, len(all_items)))]

    rows = []
    for c_idx, c in enumerate(range(1000, 1000 + n_customers)):
        cluster = int(cust_cluster[c_idx])
        n_orders = int(rng.integers(2, 13))  # 2-12 inclusive
        for _ in range(n_orders):
            day = int(rng.integers(0, 600))
            invoice = str(rng.integers(500000, 600000))
            # 1-4 distinct items per basket to give item2item co-occurrence signal
            n_lines = int(rng.integers(1, 5))
            basket_items: set[str] = set()
            attempts = 0
            while len(basket_items) < n_lines and attempts < 20:
                basket_items.add(_draw_item(cluster))
                attempts += 1
            for item in basket_items:
                rows.append({
                    "Invoice": invoice,
                    "StockCode": item,
                    "Description": "x",
                    "Quantity": int(rng.integers(1, 6)),
                    "InvoiceDate": start + pd.Timedelta(days=day),
                    "Price": float(round(rng.uniform(0.5, 20), 2)),
                    "Customer ID": c,
                    "Country": "United Kingdom",
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if RAW.exists():
        df = pd.read_parquet(RAW)
        top_customers = df["Customer ID"].value_counts().head(200).index
        df = df[df["Customer ID"].isin(top_customers)]
    else:
        print("Raw parquet absent; synthesising a sample.", flush=True)
        df = _synth()
    clean_transactions(df).to_parquet(OUT, index=False)  # store CLEANED for fast tests
    print(f"Wrote sample with {len(pd.read_parquet(OUT)):,} cleaned rows", flush=True)


if __name__ == "__main__":
    main()
