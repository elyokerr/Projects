# Data

This project commits a small sample so it clones, tests, and runs the full evaluation with no download. Working data (`raw/`, `processed/`) is gitignored.

## Committed sample

`tests/fixtures/sample_transactions.parquet` holds cleaned transactions with latent customer-cluster structure, built by `scripts/build_sample.py`. It is synthetic when the full dataset is absent, so the model ladder shows real collaborative signal (item-to-item and ALS beat popularity) without needing the download.

## Online Retail II (the full dataset)

The real data is the Online Retail II dataset from the UCI Machine Learning Repository: about 1.07 million transaction rows for a real UK-based online retailer between December 2009 and December 2011.

- Source: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- Download and cache it as parquet:
  ```bash
  python scripts/download_data.py
  ```
  This writes `data/raw/online_retail_II.parquet`.
- Rebuild the committed sample from the real data (top customers):
  ```bash
  python scripts/build_sample.py
  ```
  Without the raw parquet present, this regenerates the synthetic structured sample instead.
- Train the full pipeline and write artifacts to `models/`:
  ```bash
  python scripts/train_all.py
  ```

The data is used under the UCI Machine Learning Repository terms.
