# Data

The contents of `raw/`, `interim/`, `processed/`, and `external/` are gitignored. The folder
structure is committed (via `.gitkeep` files) but the data files themselves are not.

## Cookie Cats dataset

The real-data notebooks (`notebooks/01_eda_cookie_cats.ipynb` and
`notebooks/03_frequentist_bayesian.ipynb`) analyse the **Mobile Games A/B Testing — Cookie Cats**
dataset published on Kaggle by yufengsui.

- **Source:** [https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing-cookie-cats](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing-cookie-cats)
- **Rows:** ~90,189
- **Columns:**

| Column | Type | Description |
|---|---|---|
| `userid` | int | Unique player identifier |
| `version` | str | `gate_30` = control group; `gate_40` = treatment group |
| `sum_gamerounds` | int | Number of game rounds played in the first 14 days after install |
| `retention_1` | bool | Did the player return 1 day after install? |
| `retention_7` | bool | Did the player return 7 days after install? |

## How to get the data

**Option A — Kaggle CLI (recommended):**

```bash
kaggle datasets download -d yufengsui/mobile-games-ab-testing-cookie-cats
unzip mobile-games-ab-testing-cookie-cats.zip -d data/raw/
```

**Option B — manual download:**

1. Log in to Kaggle and go to the dataset page above.
2. Click **Download** and save the zip, then extract `cookie_cats.csv`.
3. Place the file at `data/raw/cookie_cats.csv`.

## Notes

- `data/raw/` is gitignored, so `cookie_cats.csv` is never committed to the repository.
- The simulator-based notebooks (`notebooks/04_cuped_variance_reduction.ipynb`,
  `notebooks/05_sequential_testing.ipynb`, `notebooks/06_simulation_validation.ipynb`) generate
  all their own data at runtime and **do not require the Cookie Cats download**.
- Notebooks `01_eda_cookie_cats.ipynb` and `03_frequentist_bayesian.ipynb` (the real-data case
  study) require the CSV; they guard on its absence and print a download reminder if it is missing.
