# Data

This project uses a single raw dataset: **`UJ_datatask_prices.csv`** — a panel of home-insurance quotes for 18,720 unique scenarios across seven insurers (385,025 raw rows).

## Where the file goes

```
data/raw/UJ_datatask_prices.csv
```

## Why it's not in the repo

The raw CSV is approximately **160 MB**, which exceeds GitHub's 100 MB hard limit per file. It is excluded from version control via the repo-wide `.gitignore` and the project's own `.gitignore`.

## How to obtain it

This is a proprietary dataset shared as part of the Urban Jungle data task. If you have a copy, place it at:

```
urban-jungle-price-estimator/data/raw/UJ_datatask_prices.csv
```

The first run of the notebook or Streamlit app will detect it, train the model, and cache the trained bundle to `models/uj_price_estimator_bundle.joblib`. The cached bundle (~19 MB) **is** committed to the repo, so the app can launch without re-training as long as the bundle is present.

## Folder structure

| Folder | Contents |
|---|---|
| `raw/` | Original input data (gitignored) |
| `interim/` | Intermediate transformations (not used by this project) |
| `processed/` | Final feature sets (not used — the notebook builds features in-memory) |
| `external/` | Third-party reference data (not used) |
