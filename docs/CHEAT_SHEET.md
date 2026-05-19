# Urban Jungle Price Estimator — Project Cheat Sheet

> A one-page summary for hiring managers and non-technical stakeholders.

---

## What is this project?

Urban Jungle is a home insurance aggregator. When a customer requests a quote, multiple insurers on their panel return different prices. This project builds a **machine learning model that predicts the cheapest price a customer will be offered**, based on their personal and property details — before the insurers are even contacted.

---

## Why does it matter?

- **Marketing**: Tell customers "from £X for your profile" before they fill in a full form.
- **Customer experience**: Set price expectations early so customers aren't surprised.
- **Business intelligence**: Understand which customer types get good deals and which don't.
- **Efficiency**: Pre-screen quotes without making live insurer calls.

---

## What data was used?

- **385,000 individual price quotes** from 7 insurers for 18,720 unique customer scenarios.
- Each scenario is defined by just **6 factors**: age, occupation, number of bedrooms, postcode, alarm type, and accidental damage cover.
- The target: the **cheapest quote** available for each scenario.

---

## What was the approach?

1. **Cleaned and explored the data** — checked quality, identified which columns actually matter (6 out of 39), understood pricing patterns.
2. **Tested four models** of increasing sophistication — from a simple average to advanced gradient boosting — each one had to prove it was better than the last.
3. **Validated honestly** — tested not just on random holdouts but also by holding out entire postcodes and occupations to see how the model handles genuinely new situations.
4. **Added prediction intervals** — instead of just "your quote will be £130," the model says "between £127 and £134 with 80% confidence."
5. **Made it explainable** — used SHAP to show *why* each customer gets a particular price.
6. **Made it deployable** — saved the trained model and built a ready-to-use prediction function.

---

## What were the results?

| Metric | Value |
|---|---|
| Average prediction error (known postcodes) | **£0.67** (~0.4% off) |
| Average prediction error (new postcodes) | ~£24 (geography is the dominant factor) |
| Model accuracy (R-squared) | **99.9%** on known scenarios |
| Prediction confidence band | 80% of actual prices fall within the predicted range |

**In plain English:** For postcodes the model has seen, it predicts the cheapest available quote to within 67 pence on average. For entirely new postcodes, performance drops significantly — this is expected and tells us exactly where to invest next (external geographic data like crime rates and flood risk).

---

## What are the key business insights?

1. **Location is everything.** Postcode alone explains 43% of the price difference between customers.
2. **Property size is second.** A 5-bedroom house costs roughly £61 more than a 1-bedroom.
3. **Only 6 factors drive pricing.** The other 33 data fields in the dataset carry zero information — a lean pipeline needs only these 6.
4. **Security features matter more than they appear.** Alarm type looks minor on its own but significantly affects pricing in certain postcodes.
5. **The model is unbiased.** It doesn't systematically over- or under-predict at any price level.

---

## What tools were used?

- **Python** with scikit-learn (industry-standard machine learning library)
- **HistGradientBoosting** — the winning model type, known for strong performance on tabular data
- **SHAP** — for explainable AI (showing why each prediction was made)
- Runs in **Google Colab** (free, browser-based) or any local Python environment

---

## What would come next?

| Priority | Next step | Expected impact |
|---|---|---|
| 1 | Interactive demo (Streamlit app) | Stakeholders can try scenarios themselves |
| 2 | Add external data (crime rates, flood risk) | Improve predictions for new postcodes |
| 3 | Production API | Integrate into the live customer journey |
| 4 | Drift monitoring | Alert when insurer pricing changes |

---

## Technical quality markers

- All results are **reproducible** (fixed random seeds, pinned library versions).
- Three validation strategies ensure results are **honest**, not just optimistic.
- The model is **saved and ready to deploy** — not just a notebook experiment.
- Every decision in the notebook is paired with its **reasoning** — auditable end-to-end.

---

*This project demonstrates: data analysis, feature engineering, model selection, honest validation, explainability, and production-readiness — applied to a real insurance pricing problem.*
