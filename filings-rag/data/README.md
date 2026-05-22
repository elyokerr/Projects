# Data

The contents of `raw/`, `interim/`, `processed/`, and `external/` are gitignored by the repo-wide `.gitignore`. The folder structure is committed (via `.gitkeep` files) but the data files themselves are not.

## Where to put what

| Folder | Use for |
|---|---|
| `raw/` | The original, immutable input data. Never edit these files. |
| `interim/` | Intermediate transformations — outputs of cleaning, joins, reshaping. |
| `processed/` | Final, model-ready feature sets. |
| `external/` | Third-party data (reference tables, public datasets, lookup files). |

## How to get the data

Document here where the dataset comes from — a download link, a Kaggle URL, an internal source, etc. Anyone cloning the repo should be able to recreate the `raw/` files from this README.

Example:

> The raw data is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in `data/raw/`.
