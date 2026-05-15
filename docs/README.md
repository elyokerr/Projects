# Urban Jungle — Lowest-Price Home Insurance Quote Estimator

> **Take-home data task — Round 2 interview submission**
> Predicting the **lowest** premium a panel of insurers would offer for a given home-insurance quote, from the customer's input circumstances alone.

---

## Table of contents

1. [The business problem](#1-the-business-problem)
2. [Dataset](#2-dataset)
3. [Tech stack](#3-tech-stack)
4. [Project structure](#4-project-structure)
5. [Methodology — step by step](#5-methodology--step-by-step)
6. [Results](#6-results)
7. [Visualisations](#7-visualisations)
8. [Key findings & business takeaways](#8-key-findings--business-takeaways)
9. [Assumptions](#9-assumptions)
10. [Limitations & next steps](#10-limitations--next-steps)
11. [How to reproduce](#11-how-to-reproduce)

---

## 1. The business problem

Urban Jungle operates as an aggregator: for any given customer, several insurers on the panel return quotes, and the customer is interested primarily in the **cheapest** offer available. Being able to *estimate the lowest premium up-front*, before triggering live underwriting calls, is commercially valuable because it:

- enables **price-led marketing** ("from £X for your circumstances…");
- supports **product analytics** — understanding which customer segments are well-served by the panel and which aren't;
- drives **funnel optimisation** — flagging customers likely to abandon if the genuine market price is high.

The task is therefore framed as a **regression problem**: given a subset of input circumstances, predict `min(TotalAmountPayable)` over all insurer / payment-method offerings for that quote.

---

## 2. Dataset

| Property | Value |
|---|---|
| File | `UJ_datatask_prices.csv` |
| Rows | 385,025 raw quotes |
| Unique `QUOTEID`s (input scenarios) | **18,720** |
| Insurers on panel | 7 (`A`–`G`) |
| Quotes per QuoteID | mean ≈ 21, range 5 – 35 |
| Insurers per QuoteID | mean ≈ 4, range 1 – 7 |
| Product | Home Insurance (single product) |
| Payment methods | DirectDebit, CreditCard, DebitCard |
| Payment frequencies | Annual, Monthly, Monthly2MonthsFree |
| Target column | `TotalAmountPayable` (column AB) |

**Important structural insight**: each `QUOTEID` is a *unique customer/property scenario*, and each row is one quote offered for that scenario. The same customer can attract many quotes (across insurer × payment method × frequency). Our target is the **minimum** of those offers.

---

## 3. Tech stack

| Layer | Tool |
|---|---|
| Language | **Python 3.12** |
| Notebook | Jupyter / **Google Colab** |
| Storage | **Google Drive** (mounted via `google.colab.drive`) for plot artefacts |
| Data wrangling | **pandas 2.2**, **numpy 2.0** |
| Modelling | **scikit-learn 1.6** — `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `DummyRegressor` |
| Pipelines | `Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler` |
| Validation | `KFold` (5-fold cross-validation), `train_test_split` |
| Diagnostics | `permutation_importance`, residual analysis |
| Plotting | **matplotlib 3.10**, **seaborn 0.13** |

The full set of versions is printed in the notebook's first cell for reproducibility.

---

## 4. Project structure

```
Urabn Jungle Task/
├── README.md                       <-- you are here
├── README.txt                      <-- original task brief
├── UJ_datatask_prices.csv          <-- raw quote data (385k rows)
├── UJ_price_estimator.ipynb        <-- main analysis notebook
├── target_distribution.png         <-- exported plot (also in Drive)
├── univariate_boxplots.png
├── pred_vs_actual_residuals.png
└── feature_importance.png
```

Plots are written to `/content/drive/MyDrive/Urban_Jungle_Task/plots/` during execution and downloaded back to the repo root for embedding here.

---

## 5. Methodology — step by step

The notebook is organised as a deliberate progression: **understand the data → reduce it to its real signal → model it → defend the model.** Each step's rationale is given below.

### 5.1 Structural EDA

Before touching a model we mapped the *shape* of the data: how many quotes per QuoteID, how many insurers compete on each, the distribution of `TotalAmountPayable`. **Why this first?** Mis-framing the problem (e.g. modelling individual-quote price instead of the per-QuoteID minimum) would invalidate every downstream choice.

### 5.2 Input-feature triage

The README declares columns from `AD` onward as input circumstances — 39 columns. We counted the unique values of each:

> **Of those 39 columns, 31 are constants** (a single value across the entire 385k rows) and one (`INDUSTRY`) is perfectly **1-to-1 with `OCCUPATION`** — therefore redundant.

That leaves **6 truly informative inputs**:

| Feature | Cardinality | Type |
|---|---|---|
| `DOB` (→ `AGE`) | 4 distinct ages | Numeric |
| `OCCUPATION` | 13 categories | Categorical |
| `NUMBEDROOMS` | 3 (One / Three / Five) | Ordinal |
| `INSUREDPOSTCODE` | 30 postcodes | Categorical |
| `ACCIDENTALCONTENTS` | 2 (True/False) | Binary |
| `ALARMTYPE` | 2 (NoAlarm/NonMaintained) | Binary |

The product **4 × 13 × 3 × 30 × 2 × 2 = 18,720** matches the unique-QuoteID count *exactly* — a clean sanity check that these six features fully define a scenario.

> **Why this matters commercially:** an honest readout of "we're using 6 features" is infinitely more defensible to a reviewer than a vanity readout of "we're using 39 features" when 33 of them are noise.

### 5.3 Target construction

Collapsed the 385k-row quote table into an **18,720-row per-QuoteID dataset**, one row per unique scenario, with `target = min(TotalAmountPayable)`. This is the supervised problem the README asks us to solve.

### 5.4 Univariate effect analysis

For each of the 6 features we measured (a) the spread of the per-group mean target and (b) the R² of using that group mean as a one-feature predictor:

| Feature | Spread of group mean | Univariate R² |
|---|---:|---:|
| INSUREDPOSTCODE | £176 | 0.43 |
| NUMBEDROOMS | £61 | 0.26 |
| ALARMTYPE | £31 | 0.10 |
| ACCIDENTALCONTENTS | £16 | 0.03 |
| AGE | £18 | 0.02 |
| OCCUPATION | £9 | 0.01 |

This guides modelling: **postcode and bedrooms dominate the marginal signal**, but the others may still help in interaction.

### 5.5 Feature engineering

Encoding choices were kept *minimal and defensible* — over-engineering would only hurt interpretability:

- **`DOB` → `AGE`** (integer years vs reference date 2026-05-15).
- **`NUMBEDROOMS`** → ordinal map `{One:1, Three:3, Five:5}` so a linear model spends a single coefficient instead of two dummies.
- **`ACCIDENTALCONTENTS`, `ALARMTYPE`** → 0/1 binaries.
- **`OCCUPATION`** → one-hot (13 dummies — small, no curse-of-dimensionality risk).
- **`INSUREDPOSTCODE`** → both the **full postcode** (one-hot) *and* a **postcode area** prefix (e.g. `N`, `SW`, `HA`) for tree-based models to choose from.

> **Why no target encoding?** With 30 postcodes × ~600 QuoteIDs each, simple one-hot has ample statistical support. Target encoding would risk leakage if not folded carefully — added complexity for no expected gain.

### 5.6 Train / test protocol

- **80/20 split on QuoteIDs** (each QuoteID is one row in the modelling frame, so a vanilla `train_test_split` already gives a clean holdout).
- **5-fold cross-validation on the train set** for model comparison (provides both a point and a stability estimate).
- **Test set touched only at the very end** to avoid optimistic bias.

**Evaluation metrics** — three complementary views:
- **MAE (£)** — easy for non-technical stakeholders to interpret;
- **RMSE (£)** — penalises large misses (matters if a single £200 miss is worse than four £50 misses);
- **MAPE (%)** — relative error, useful because the target spans an order of magnitude;
- **R²** — share of variance explained; convenient for cross-model comparison.

### 5.7 Modelling progression

Four models were fit in increasing order of complexity. **Each step had to *justify* its complexity by beating the previous on cross-validated MAE.**

1. **Baseline — `DummyRegressor(mean)`** sets the floor.
2. **Ridge regression** with one-hot encoding — interpretable, fast, defensible.
3. **Random Forest** (400 trees, `min_samples_leaf=2`) — captures interactions automatically.
4. **HistGradientBoosting** (400 iterations, `lr=0.05`, native categorical support) — the modern GBM workhorse, no one-hot needed for categoricals.

### 5.8 Diagnostics

For the best model we generated:
- **Predicted-vs-actual scatter** + **residual plot** (bias / heteroscedasticity check),
- **Permutation feature importance** (model-agnostic, defensible to non-ML reviewers),
- **Per-postcode error breakdown** (does any region break the model?).

---

## 6. Results

### Headline numbers (test set, n = 3,744)

| Model | MAE (£) | RMSE (£) | MAPE (%) | R² | 5-fold CV MAE |
|---|---:|---:|---:|---:|---:|
| Baseline (mean) | 35.64 | 49.14 | 22.88 | -0.00 | — |
| **Ridge (one-hot)** | 12.16 | 20.04 | 7.45 | 0.834 | £11.97 ± 0.21 |
| **Random Forest** | 4.12 | 6.66 | 2.29 | 0.982 | £4.25 ± 0.07 |
| **HistGradientBoosting** ★ | **1.46** | **2.57** | **0.86** | **0.997** | **£1.46 ± 0.03** |

★ Selected as the production model.

The **HistGradientBoosting model predicts the lowest panel quote within £1.46 on average** — i.e. less than 1 % off in MAPE terms — with a tight 5-fold CV band of ±£0.03 (very stable). Residuals are centred on zero (mean +£0.10) and show no systemic bias across the price range.

---

## 7. Visualisations

### 7.1 Target distribution

![Target distribution](target_distribution.png)

*Left*: raw `TotalAmountPayable` across all 385k quote rows — long-tailed, ranging £90 → £1,070. *Right*: the per-QuoteID **minimum**, our actual target — substantially compressed (£90 → ~£600) because by definition we always pick the cheapest insurer. Modelling on the raw scale is fine; the distribution is well-behaved without needing a log transform.

### 7.2 Univariate effect of each feature

![Univariate boxplots](univariate_boxplots.png)

The most visible drivers are **postcode** (huge between-group variation) and **bedrooms** (clear monotonic increase with property size). Age, accidental-contents and alarm type each shift the median by a small but consistent amount; occupation has a weak marginal effect on its own.

### 7.3 Predicted vs actual & residuals — HistGradientBoosting

![Predicted vs actual + residuals](pred_vs_actual_residuals.png)

*Left*: predictions hug the y = x line tightly across the full price range. *Right*: residuals scatter around zero with no fan-out or curvature → the model is **unbiased and homoscedastic**. The few outliers in the right tail are the same QuoteIDs whose lowest available quote is unusually high — natural noise in insurer pricing.

### 7.4 Permutation feature importance

![Feature importance](feature_importance.png)

Permutation importance asks "if I shuffle this column, how much does test MAE worsen?" — a model-agnostic, leakage-safe view.

- **`INSUREDPOSTCODE` (~£30)** and **`BEDROOMS_N` (~£26)** dominate, matching the univariate intuition.
- **`ALARM_BIN` (~£15)** is *much* more important here than its univariate R² (0.10) suggested — a clear sign the model uses it in **interaction** with other features (e.g. specific high-risk postcodes).
- **`POSTCODE_AREA` (~£0)** contributes nothing on top of `INSUREDPOSTCODE` — fully subsumed. We'd drop it in production for a leaner model.

---

## 8. Key findings & business takeaways

1. **Six features are enough.** The other 33 columns labelled as "inputs" carry no signal in this dataset. Honest feature triage saves the model — and the engineering pipeline that would feed it — from carrying 5× the columns it actually needs.
2. **Postcode and property size dominate**, exactly as insurance underwriting theory predicts. Geography (crime / theft / flood risk) and size of contents-at-risk are the two biggest pricing levers.
3. **Security and cover toggles matter more in combination than in isolation.** ALARM and ACCIDENTAL-CONTENTS look weak on their own but contribute meaningfully to the joint model — useful when designing optional add-ons in the customer journey.
4. **A simple model is already very good.** Even Ridge regression cuts the baseline MAE from £35 to £12 — a Pareto-efficient choice if interpretability is a hard regulatory requirement. The jump to gradient boosting buys another order of magnitude (£12 → £1.46) and is worth it if pure accuracy wins.
5. **The residuals are clean.** No systematic over- or under-prediction across the price range — meaning the model is fit for use *as a quote pre-screen* without further calibration.

---

## 9. Assumptions

- **`TransactionCharge` (column AC) is a quote *output*, not an input** — including it would leak target information.
- **Reference date for age = 2026-05-15** (today). Since DOB values are uniformly `01/01`, the choice only adds an integer constant; it doesn't shift relative model behaviour.
- **No missing-data handling required** — every column has either zero nulls or is a constant we drop.
- **The lowest *available* price is what we model** — we do not condition on which insurer or payment method delivered it. The README defines the target this way.
- **The 7 insurers, 30 postcodes and 13 occupations represent the full universe for this exercise** — the model is not expected to generalise to *new* postcodes or occupations.

---

## 10. Limitations & next steps

- **Geographic generalisation is untested.** With only 30 postcodes, a leave-one-postcode-out evaluation would tell us how the model would extend to a new region. Worth running before any geographic expansion.
- **External signals not used.** Crime rate, flood-risk index, and rebuild cost per postcode are almost certainly the highest-leverage next features.
- **Static model — pricing drifts.** Insurer pricing surfaces change quarterly; in production we'd schedule periodic re-training and monitor MAE drift.
- **Right-tail edge cases.** A small number of QuoteIDs in the right tail (predicted £400+) show wider residuals — a quantile regression head, or a separate model for the top 5 % of prices, would tighten this.
- **No hyper-parameter sweep was performed.** Modest-tuning was deliberate (the residual variance is dominated by genuine insurer-side noise, not by model capacity). A `GridSearchCV` round would likely shave another 10 – 20 % off MAE.

---

## 11. How to reproduce

### Option A — Google Colab (matches the executed environment)

1. Upload `UJ_price_estimator.ipynb` and `UJ_datatask_prices.csv` to a folder named `Urabn Jungle Task` (note the spelling) inside your Google Drive's `MyDrive` root.
2. Open the notebook in Colab.
3. Run all cells. The first cell mounts Drive and creates `MyDrive/Urban_Jungle_Task/plots/` for image artefacts.
4. Total runtime ≈ 3 – 4 minutes (RandomForest is the slowest cell).

### Option B — Local Jupyter

```bash
pip install pandas==2.2.* numpy==2.0.* scikit-learn==1.6.* matplotlib==3.10.* seaborn==0.13.* jupyter
jupyter notebook UJ_price_estimator.ipynb
```

Comment out the Colab `drive.mount(...)` block and point `SAVE_DIR` to a local folder.

---

## 12. Closing note

The submission deliberately privileges **clarity of reasoning** over raw model performance. Every modelling decision in the notebook is paired with the *why*; every metric reported is paired with the alternatives we considered. The ambition was to produce a notebook a senior reviewer could read end-to-end and understand not just *what* was built, but *why each choice was made* — exactly as the task brief requested.
