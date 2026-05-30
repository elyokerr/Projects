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


def _synth(n_customers=120, n_items=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2010-01-01")
    for c in range(1000, 1000 + n_customers):
        n_orders = rng.integers(2, 12)
        for _ in range(n_orders):
            day = int(rng.integers(0, 600))
            item = f"SKU{rng.integers(0, n_items):03d}"
            rows.append({
                "Invoice": str(rng.integers(500000, 600000)),
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
