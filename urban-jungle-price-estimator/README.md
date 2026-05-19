# Lowest-Price Home Insurance Quote Estimator

> **Task**
> Predicting the **lowest** premium a panel of insurers would offer for a given home-insurance quote, from the customer's input circumstances alone.

> **Interactive demo** - a Streamlit web app (`app/app.py`) ships with the project. See [Section 11 - Interactive web app](#11-interactive-web-app) for a one-command launch.

> **Reproducible** - clone the repo, place the raw CSV in `data/raw/`, install dependencies, and the notebook or app runs end-to-end with no path edits. See [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) for a step-by-step guide.

---

## Table of contents

1. [The business problem](#1-the-business-problem)
2. [Dataset](#2-dataset)
3. [Tech stack](#3-tech-stack)
4. [Project structure](#4-project-structure)
5. [Methodology - step by step](#5-methodology--step-by-step)
6. [Results](#6-results)
7. [Key findings & business takeaways](#7-key-findings--business-takeaways)
8. [Assumptions](#8-assumptions)
9. [Limitations & next steps](#9-limitations--next-steps)
10. [How to reproduce](#10-how-to-reproduce)
11. [Interactive web app](#11-interactive-web-app)
12. [v1 → v2 changes](#v1--v2-changes)

---

## 1. The business problem

The company operates as an aggregator: for any given customer, several insurers on the panel return quotes, and the customer is interested primarily in the **cheapest** offer available. Being able to *estimate the lowest premium up-front*, before triggering live underwriting calls, is commercially valuable because it:

- enables **price-led marketing** ("from £X for your circumstances…");
- supports **product analytics** - understanding which customer segments are well-served by the panel and which aren't;
- drives **funnel optimisation** - flagging customers likely to abandon if the genuine market price is high.

The task is framed as a **regression problem**: given a subset of input circumstances, predict `min(TotalAmountPayable)` over all insurer / payment-method offerings for that quote.

---

## 2. Dataset

| Property | Value |
|---|---|
| File | `UJ_datatask_prices.csv` |
| Rows | 385,025 raw quotes |
| Unique `QUOTEID`s (input scenarios) | **18,720** |
| Insurers on panel | 7 (`A`–`G`) |
| Quotes per QuoteID | mean ≈ 21, range 5–35 |
| Insurers per QuoteID | mean ≈ 4, range 1–7 |
| Product | Home Insurance (single product) |
| Payment methods | DirectDebit, CreditCard, DebitCard |
| Payment frequencies | Annual, Monthly, Monthly2MonthsFree |
| Target column | `TotalAmountPayable` (column AB) |
| Data quality | Zero nulls, zero duplicate rows |

**Structural insight**: each `QUOTEID` is a *unique customer/property scenario*, and each row is one quote offered for that scenario. Our target is the **minimum** of those offers per QuoteID.

---

## 3. Tech stack

| Layer | Tool | Pinned version |
|---|---|---|
| Language | **Python** | 3.10 – 3.13 |
| Notebook | **Jupyter** / **Google Colab** | jupyter ≥ 1.1, ipykernel ≥ 6.29 |
| Numerical core | **NumPy** | ≥ 2.0, < 2.3 |
| Data wrangling | **pandas** | ≥ 2.2, < 2.4 |
| Scientific stack | **SciPy** | ≥ 1.13, < 1.16 |
| Modelling | **scikit-learn** - `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `DummyRegressor` | ≥ 1.6, < 1.8 |
| Validation | `KFold`, `GroupKFold`, `RandomizedSearchCV`, `train_test_split` | (scikit-learn) |
| Explainability | **SHAP**, `permutation_importance` | shap ≥ 0.46, < 0.50 |
| Model serialisation | **joblib** | ≥ 1.4, < 1.6 |
| Static plots | **matplotlib**, **seaborn** | matplotlib ≥ 3.10, < 3.12; seaborn ≥ 0.13, < 0.15 |
| Interactive dashboard | **Streamlit** | ≥ 1.36, < 2.0 |
| Interactive charts | **Plotly** | ≥ 5.22, < 6.0 |

Every dependency is pinned in `requirements.txt`. The runtime versions are also
printed by the notebook's first cell so reviewers can confirm reproducibility.

---

## 4. Project structure

```
urban-jungle-price-estimator/
├── README.md                                  ← You are here
├── requirements.txt                           ← Pinned Python dependencies
├── .gitignore                                 ← Project-specific ignores
│
├── notebooks/
│   └── UJ_price_estimator.ipynb               ← Full analysis + modelling pipeline
│
├── app/
│   ├── app.py                                 ← Streamlit web app
│   ├── run_app.bat                            ← Windows launcher
│   └── run_app.sh                             ← macOS / Linux launcher
│
├── models/
│   └── uj_price_estimator_bundle.joblib       ← Trained point + quantile models
│
├── data/
│   ├── README.md                              ← Where to put the raw CSV
│   └── raw/                                   ← UJ_datatask_prices.csv (gitignored)
│
└── docs/
    └── HOW_TO_RUN.md                          ← Full run-instructions guide
```

---

## 5. Methodology - step by step

The notebook follows a deliberate progression: **understand the data → reduce it to its real signal → model it → validate honestly → explain it → make it deployable.**

### 5.1 Data quality analysis

Before any modelling: check dtypes, nulls, duplicates, cardinality per column, and memory footprint. Result: zero nulls, zero duplicates, 44 constant columns out of 67 - the dataset is a curated extract, not raw production data.

### 5.2 Structural EDA

Map the shape of the data: how many quotes per QuoteID, how many insurers compete, panel coverage per insurer. This confirms the problem framing - each QuoteID is a unique scenario, and the task is to predict the cheapest quote across all insurers.

### 5.3 Target deep dive

Analyse the distribution of `min(TotalAmountPayable)`: skew (+2.19), kurtosis (9.72), Q-Q plot against normal, and a log-transform test. Decision: model on raw £ scale - tree models are scale-invariant, and raw £ keeps the output directly interpretable. IQR outlier scan finds 3.1% above the upper fence, all genuine high-risk scenarios (5-bedroom properties in expensive postcodes).

### 5.4 Input-feature triage

Of 39 declared input columns, **31 are constants**, 1 (`INDUSTRY`) is redundant with `OCCUPATION`, and 1 (`TransactionCharge`) is a quote output that would leak target information. That leaves **6 effective features**:

| Feature | Cardinality | Type |
|---|---|---|
| `DOB` (→ `AGE`) | 4 distinct ages | Numeric |
| `OCCUPATION` | 13 categories | Categorical |
| `NUMBEDROOMS` | 3 (One / Three / Five) | Ordinal |
| `INSUREDPOSTCODE` | 30 postcodes | Categorical |
| `ACCIDENTALCONTENTS` | 2 (True/False) | Binary |
| `ALARMTYPE` | 2 (NoAlarm/NonMaintained) | Binary |

The product **4 × 13 × 3 × 30 × 2 × 2 = 18,720** matches the unique-QuoteID count exactly - these 6 features fully define every scenario.

### 5.5 Univariate and bivariate EDA

Spearman correlations, per-feature univariate R², and 2-way interaction heatmaps. Key finding: postcode alone explains 43% of variance, bedrooms 26%, and combining them yields more than the sum - confirming interaction effects that tree models will capture.

### 5.6 Feature engineering

- **`DOB` → `AGE`** (integer years)
- **`NUMBEDROOMS`** → ordinal `{One:1, Three:3, Five:5}`
- **`ACCIDENTALCONTENTS`, `ALARMTYPE`** → 0/1 binaries
- **`OCCUPATION`, `INSUREDPOSTCODE`** → native categorical for HGB, one-hot for Ridge/RF
- **`POSTCODE_OUTWARD`** (e.g. `N16`, `SW7`) - the standard UK insurance geo unit, new in v2
- **`POSTCODE_AREA`** (e.g. `N`, `SW`) - coarse geographic grouping

### 5.7 Train/test protocol - three splits

| Split | What it tests | Why |
|---|---|---|
| Random 80/20 | Interpolation within the sampled grid | Comparable to v1; optimistic baseline |
| **GroupKFold by postcode** | Geographic generalisation | Model has never seen this postcode |
| **GroupKFold by occupation** | Customer-type generalisation | Model has never seen this occupation |

The gap between random-split and group-split MAE is the most commercially important number in the notebook.

### 5.8 Modelling progression

Four models in increasing complexity, each justified by beating the previous:

1. **Baseline** - `DummyRegressor(mean)` sets the floor
2. **Ridge regression** - interpretable, fast, defensible
3. **Random Forest** - captures interactions automatically
4. **HistGradientBoosting** - native categorical support, the modern GBM workhorse

### 5.9 Hyperparameter tuning

`RandomizedSearchCV` (25 iterations) over HGB's key hyperparameters: `max_iter`, `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`. Best found: 800 iterations, lr=0.05, 127 leaf nodes, L2=1.0.

### 5.10 Diagnostics

- Predicted-vs-actual scatter + residual plot (bias check)
- Permutation feature importance (model-agnostic)
- **SHAP summary plot** (per-prediction explanations)
- Error breakdown by postcode and bedroom segment

### 5.11 Quantile regression

Three HGB heads at q = {0.1, 0.5, 0.9} produce an 80% prediction band. Empirical coverage: 75.4%, mean band width: £15.83.

### 5.12 Productionisation

Trained point-estimate model + quantile heads serialised with `joblib` into a single bundle. A `predict_lowest_price()` function takes a customer scenario dict and returns `{point, q10, q50, q90}`.

---

## 6. Results

### Random 80/20 split - test set (n = 3,744)

| Model | MAE (£) | RMSE (£) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Baseline (mean) | 35.64 | 49.14 | 22.88 | -0.00 |
| Ridge (one-hot) | 12.16 | 20.04 | 7.45 | 0.834 |
| Random Forest | 4.13 | 6.66 | 2.29 | 0.982 |
| HGB (default) | 1.46 | 2.57 | 0.86 | 0.997 |
| **HGB (tuned)** ★ | **0.67** | **1.45** | **0.38** | **0.999** |

★ Selected as the production model.

### Honest validation - 5-fold CV MAE across splitting regimes

| Model | Random KFold | GroupKFold postcode | GroupKFold occupation |
|---|---:|---:|---:|
| Ridge | £12.02 | £25.91 | £12.68 |
| Random Forest | £4.07 | £24.07 | £6.67 |
| **HGB (tuned)** | **£0.77** | **£23.75** | **£7.18** |

**The key insight:** on a random split, the tuned HGB is accurate to within £0.67. But if we hold out an entire postcode the model has never seen, the error rises to ~£24 — because geography is the dominant pricing factor and the model cannot extrapolate to unseen locations. This is honest, not a failure; it tells us exactly where external data (crime rate, flood risk) would have the highest leverage.

### Quantile regression

| Metric | Value |
|---|---|
| Empirical coverage of [q10, q90] | 75.4% (target 80%) |
| Mean band width | £15.83 |

### Worked examples

| Scenario | Point estimate | 80% band |
|---|---:|---|
| 30yo, 1-bed, no alarm, no accidental, D78, N65TX | £115.81 | £115.03 – £117.08 |
| 50yo, 5-bed, alarm, accidental, A01, SW71AA | £179.53 | £176.58 – £181.43 |
| 40yo, 3-bed, no alarm, accidental, E09, HA19NA | £129.77 | £127.83 – £129.34 |

---

## 7. Key findings & business takeaways

1. **Six features are enough.** 33 of 39 declared input columns carry no signal. Honest triage avoids a bloated pipeline.
2. **Postcode and property size dominate** - geography (crime/theft/flood risk) and contents-at-risk are the two biggest pricing levers, exactly as insurance underwriting theory predicts.
3. **Security and cover toggles matter more in combination than alone.** Alarm type looks weak by itself (R² = 0.10) but contributes significantly in interaction with postcode - useful for designing optional add-ons.
4. **A simple model is already good.** Ridge cuts the baseline from £35 to £12 - a viable choice under strict interpretability requirements. Gradient boosting pushes to sub-£1 error.
5. **Geographic generalisation is the real challenge.** The model excels at interpolation but cannot predict for unseen postcodes without external geographic signals.
6. **Prediction intervals add real business value.** "Between £127 and £134" is far more useful to a customer-facing UI than "£130."
7. **The model is unbiased.** Residuals centre on zero with no systematic over- or under-prediction across the price range - suitable for use as a quote pre-screen.

---

## 8. Assumptions

- **`TransactionCharge` is a quote output, not an input** - including it would leak target information.
- **Reference date for age = 2026-05-15.** DOB values are uniformly `01/01`, so the choice only adds an integer constant.
- **No missing-data handling required** - zero nulls across the entire dataset.
- **The lowest available price is what we model** - we do not condition on which insurer or payment method delivered it.
- **The 7 insurers, 30 postcodes and 13 occupations represent the full universe for this exercise** - the model is not expected to generalise to entirely new categories.

---
## 10. How to reproduce

> See [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for the fully-detailed
> step-by-step guide with troubleshooting.

### Quick start (local - recommended)

```bash
# 1. Clone the portfolio repo and enter this project
git clone https://github.com/elyokerr/Projects.git
cd Projects/urban-jungle-price-estimator

# 2. Place the raw CSV at data/raw/UJ_datatask_prices.csv
#    (see data/README.md for details)

# 3. Install dependencies (one-time)
pip install -r requirements.txt

# 4a. Run the notebook
jupyter notebook notebooks/UJ_price_estimator.ipynb

# 4b. ...or launch the interactive dashboard
streamlit run app/app.py
```

### Quick start (Google Colab)

1. Open Colab → **File → Upload notebook** → choose `notebooks/UJ_price_estimator.ipynb`.
2. In the Colab left sidebar, click the **Files** icon and upload `UJ_datatask_prices.csv`.
3. **Run all cells.** The first cell auto-detects Colab and reads the uploaded CSV directly from `/content/`. No Drive mount required.

### Using the trained model programmatically

```python
import joblib

bundle = joblib.load('models/uj_price_estimator_bundle.joblib')
# bundle contains: point_model, quantile_models, feature_cols, cat_levels, etc.

# Or use the predict_lowest_price() function defined in the notebook
```

## 11. Interactive web app

An interactive **Streamlit web app** (`app.py`) is included to demonstrate the
real-world usage of the price estimator. It simulates how a customer-facing UI
or internal pricing tool would consume the trained model.

### Features
- **Customer profile form** - age, bedrooms, postcode, occupation, alarm & accidental-cover toggles
- **Headline price + 80% confidence band** with a speedometer-style gauge
- **Quick presets** - three sample customer profiles (young renter, family home, senior in affluent area)
- **Market insights tab** - interactive charts comparing prices across bedrooms and postcodes
- **Model card** - algorithm, hyperparameters, training stats, test-set metrics
- **Honest validation panel** - shows the random-split vs group-split MAE gap

### One-command launch

**Windows** (from the project root):
```cmd
app\run_app.bat
```

**macOS / Linux** (from the project root):
```bash
./app/run_app.sh
```

**Or directly (any platform):**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### How it works

- **First run** (~30 seconds): trains the HGB point-estimate model and the three
  quantile heads from `data/raw/UJ_datatask_prices.csv`, then saves the bundle
  to `models/uj_price_estimator_bundle.joblib`.
- **Subsequent runs**: loads the cached bundle instantly via `@st.cache_resource`.
- **No internet required** after the initial `pip install`.

### Tech stack

| Component | Tool |
|---|---|
| Web framework | **Streamlit** - the industry-standard ML demo framework |
| Charts | **Plotly** - interactive gauge, band, and bar charts |
| Model serving | **joblib**-loaded scikit-learn pipeline |
| State | Streamlit's built-in `@cache_resource` decorator |

### Production path

This app simulates the customer-facing surface; a production deployment would
typically wrap the same `predict_lowest_price()` function in a **FastAPI**
service, containerise with **Docker**, and serve behind an authenticated
gateway. The Streamlit demo is the fastest route from notebook to a tangible,
stakeholder-demoable artefact.

---
