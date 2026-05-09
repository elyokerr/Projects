# Modelling Approach

> How the churn prediction model was selected, trained, evaluated, and optimised for business impact.

---

## Overview

The modelling phase takes the 45 engineered features from the data pipeline and builds a predictive system that answers two questions:

1. **Which customers are likely to churn?** — Binary classification with calibrated probabilities
2. **Why are they likely to churn?** — SHAP-based explanations for each prediction

Rather than simply training one model and reporting accuracy, the approach follows industry best practices: compare multiple algorithms, track experiments systematically, optimise for business outcomes rather than statistical metrics, and ensure every prediction is explainable.

---

## Model Selection

### Why three models?

Three algorithms were compared to find the best fit for this problem:

**Logistic Regression** — The simplest option. It's fast, fully interpretable, and serves as a baseline. If a logistic regression performs well enough, there's no need for a more complex model. In industry, simpler models are often preferred because they're easier to maintain, debug, and explain.

**Random Forest** — A moderate-complexity ensemble method. It handles non-linear relationships and feature interactions that logistic regression cannot capture. It's also robust to outliers and doesn't require extensive feature scaling.

**XGBoost** — A gradient-boosted tree algorithm that consistently wins tabular data competitions and is the most commonly used algorithm for structured business data in industry. It's more powerful than Random Forest but requires more careful tuning.

### How they were compared

All three models were trained on the same training set (80% of data) and evaluated on the same held-out test set (20%). The split was stratified — meaning the proportion of churners in both sets matches the original dataset — to prevent evaluation bias.

Every experiment was tracked using **MLflow**, which logs:

- All hyperparameters used
- All evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
- The trained model artifact itself
- The date and time of each run

This means any experiment can be reproduced or compared at any time. MLflow is the industry standard for experiment tracking and is used at companies ranging from startups to enterprises.

---

## Evaluation Metrics

### Why not just accuracy?

In this dataset, roughly 73% of customers did not churn and 27% did. A model that simply predicts "no churn" for everyone would be 73% accurate — but completely useless. This is known as the **class imbalance problem**, and it's one of the most common pitfalls in real-world ML.

Instead, the evaluation focuses on metrics that matter for imbalanced classification:

**Recall (Sensitivity)** — Of all customers who actually churned, what percentage did the model correctly identify? This is the most important metric for churn prediction because a missed churner is a lost customer.

**Precision** — Of all customers the model flagged as likely to churn, what percentage actually did? Low precision means the customer success team wastes time on false alarms.

**F1 Score** — The harmonic mean of precision and recall. Useful as a single summary metric that balances both concerns.

**AUC-ROC** — How well the model distinguishes between churners and non-churners across all possible thresholds. A score of 1.0 is perfect; 0.5 is no better than random guessing.

---

## Threshold Optimisation

### The problem with 0.5

Most classification models output a probability (e.g., "this customer has a 0.63 probability of churning"). By default, a threshold of 0.5 is used — anything above 0.5 is classified as "will churn". But 0.5 is rarely the right threshold for business problems.

### How the threshold was tuned

The threshold was optimised by considering the business cost of errors:

- **False negative** (missed churner) — The company loses the customer's monthly recurring revenue, plus the cost of acquiring a replacement customer. This is expensive.
- **False positive** (false alarm) — The customer success team reaches out to a customer who wasn't actually going to churn. This costs some staff time but causes no real harm.

Because false negatives are much more costly than false positives, the optimal threshold is lower than 0.5. This means the model is deliberately more aggressive about flagging potential churners, accepting some false alarms in exchange for catching more real ones.

The final threshold was selected by maximising the F1 score on the validation set, which provides a good balance for this cost structure.

---

## Model Explainability with SHAP

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a method from game theory that explains how much each feature contributed to a specific prediction. It answers the question: "Why did the model predict this customer would churn?"

### Global explanations

The SHAP summary plot shows which features are most important across the entire dataset and how they influence predictions:

- Features at the top of the plot have the most impact on predictions
- Red dots indicate high feature values; blue dots indicate low values
- The horizontal position shows whether the feature pushes toward churn (right) or retention (left)

For example, if "months since last login" appears at the top with red dots pushed to the right, it means customers who haven't logged in recently are strongly predicted to churn — which makes intuitive business sense.

### Individual explanations

The SHAP waterfall plot explains a single prediction. For a specific customer flagged as high-risk, it shows exactly which features pushed the prediction toward churn and by how much. This is critical for customer success teams because it tells them not just *who* to contact, but *what* to address.

For example, a waterfall plot might show that a particular customer is flagged because of declining login frequency and an expiring contract — suggesting the team should demonstrate product value and offer a renewal incentive.

---

## Revenue Impact Analysis

### Why this matters

Telling a business leader "the model has 82% recall" is meaningless to them. Translating model performance into financial terms — "the model identifies £64,000 in annual revenue at risk" — makes the value concrete and actionable.

### How it's calculated

1. Each customer has a known monthly recurring revenue (MRR)
2. The model identifies which customers are likely to churn
3. The total MRR of correctly identified churners is the "revenue at risk that can be acted on"
4. Assuming the customer success team can retain 25% of flagged customers through intervention, the estimated annual savings are calculated

This kind of business impact framing is what separates a strong data science candidate from one who only thinks in terms of model metrics.

---

## Batch Scoring

### What it does

After the model is trained and the optimal threshold is set, every customer in the database is scored with a churn probability. Based on that probability, each customer is assigned a risk level:

| Risk Level | Probability Range | Suggested Action |
|---|---|---|
| **Critical** | ≥ 0.8 | Immediate outreach from account manager |
| **High** | 0.6 – 0.8 | Proactive check-in within the week |
| **Medium** | 0.4 – 0.6 | Monitor engagement, schedule periodic review |
| **Low** | < 0.4 | No immediate action needed |

The scored customer list is saved to `data/processed/scored_customers.csv` and forms the basis for the planned Streamlit dashboard (Phase 4).

---

## Generated Plots

All visualisations are saved to `models/plots/` and include:

| Plot | What It Shows |
|---|---|
| `churn_distribution.png` | Baseline class balance (churned vs. retained) |
| `feature_correlation.png` | Heatmap of relationships between top features |
| `churn_by_contract.png` | Churn rates across contract types |
| `churn_by_tenure.png` | Churn rates across customer tenure groups |
| `churn_by_payment.png` | Churn rates by payment method |
| `roc_curve.png` | Model discrimination at all thresholds |
| `precision_recall_curve.png` | Performance on the minority class |
| `confusion_matrix.png` | Prediction breakdown at the optimised threshold |
| `shap_summary.png` | Global feature importance with direction of impact |
| `shap_bar.png` | Simplified feature importance ranking |
| `shap_waterfall.png` | Individual prediction explanation |

---

## Reproducibility

- All random seeds are fixed for reproducible results
- MLflow logs every experiment with full parameters and metrics
- The training pipeline can be re-run at any time via the notebook or command line
- Model artifacts (the trained model and metadata) are serialised with joblib for reliable loading
