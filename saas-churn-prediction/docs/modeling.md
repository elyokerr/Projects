# Modeling

This document covers how the model is trained, selected, evaluated, and explained.

## Approach

I compare three classifiers on the engineered feature set, track every run with MLflow, optimize the decision threshold for F1, and serialize the winning model with metadata that downstream services depend on.

The three models are chosen to span the typical complexity range for tabular classification:

| Model | Why it's in the comparison |
|---|---|
| Logistic Regression | Strong interpretable baseline. If a more complex model can't beat this, the added complexity isn't earning its keep. |
| Random Forest | Handles non-linearities and interactions out of the box. Less prone to overfitting than a single decision tree. |
| XGBoost | The industry workhorse for tabular data. Gradient boosting consistently wins on this kind of problem. |

All three are trained on the same train/test split (80/20, stratified, seed 42) so the comparison is apples-to-apples.

## Hyperparameters

The hyperparameters in `src/models/train.py` were chosen based on a small manual sweep and standard rules of thumb. A proper hyperparameter search (Optuna or GridSearchCV) would be the next step in a production project; for a portfolio piece, manageable defaults plus MLflow tracking lets me revisit and tune later without redoing the scaffolding.

The most important hyperparameter across all three models is the class weight / scale_pos_weight. The dataset has a 73:27 active-to-churned ratio, which would bias an unweighted model toward predicting "active" for everyone. Setting `class_weight="balanced"` on the scikit-learn models, and `scale_pos_weight=2.8` (the ratio of negatives to positives) on XGBoost, corrects for the imbalance.

## Preprocessing

`StandardScaler` is fit on the training set and applied to both train and test. Logistic regression needs scaled inputs to behave well; tree-based models don't, but I apply the same transform to all three to keep the comparison clean and to cache a single scaler artifact for the API.

The scaler is serialized to `models/scaler.joblib`. At inference time, the API checks the model name in metadata and applies the scaler only when the winning model is logistic regression.

## Cross-Validation

In addition to the train/test split, I run 5-fold stratified cross-validation on the training set with ROC-AUC as the scoring metric. This is a stability check: if a model's CV scores are tight (low standard deviation), the train/test score is trustworthy. If they're wide, the model is sensitive to which rows ended up in the training set and the test score might be optimistic.

Cross-validation results are logged to MLflow alongside the test set metrics.

## Threshold Optimization

The default sklearn classification threshold is 0.5. Choosing 0.5 implicitly says false positives and false negatives have equal cost, which is almost never true in retention contexts.

After picking the winning model by ROC-AUC, I scan the precision-recall curve and pick the threshold that maximizes F1 on the test set. F1 is the harmonic mean of precision and recall - it rewards models that catch a lot of churners without flagging too many active customers.

The optimized threshold is typically lower than 0.5 (somewhere around 0.35-0.45 depending on the run), which means the model is willing to make more false-positive predictions in exchange for catching more true churners. That trade-off is appropriate when missing a churner costs more than incorrectly flagging an active customer - which is the case in most SaaS retention contexts because customer success interventions are cheap.

A different cost structure would justify a different threshold. The training pipeline could easily be extended to accept a cost ratio and compute the threshold from it.

## Revenue Impact Estimation

After threshold optimization, I translate model performance into financial terms:

- True positives × their actual monthly revenue = MRR identified by the model
- False negatives × their actual monthly revenue = MRR missed
- MRR identified × 20% intervention success rate = monthly savings estimate
- Monthly savings × 12 = annual savings estimate

The 20% intervention success rate is a conservative assumption - industry case studies suggest 15-30% is typical for proactive retention campaigns. I deliberately picked the lower end of that range so the number sounds defensible.

The output is something like:

```
Total MRR at risk: £20,180.00
MRR identified by model: £16,500.00
Estimated monthly savings (20% retention): £3,300.00
Estimated annual savings: £39,600.00
```

This kind of framing matters for stakeholder conversations. "Recall is 0.82" is meaningless to a non-technical audience. "The model identifies £40,000 of recoverable annual revenue" is immediately actionable.

## Evaluation Plots

`src/models/evaluate.py` produces a standard set of visualizations saved to `models/plots/`:

- **ROC curve** - Classic discrimination plot. AUC is the headline number.
- **Precision-Recall curve** - More informative than ROC for imbalanced data. Shows the trade-off across all thresholds, with a horizontal line at the baseline precision (the dataset's churn rate).
- **Confusion matrix** - At the optimized threshold, not the default 0.5. Lets you read off true positives, false positives, true negatives, false negatives directly.
- **Feature importance** - Top 20 features. Uses native importance (gain for trees, absolute coefficient for linear models).

## SHAP Explanations

SHAP values are the gold standard for ML explainability. They have strong theoretical foundations (Shapley values from cooperative game theory) and produce additive, locally accurate explanations - meaning the SHAP values for a single prediction sum to the difference between the model's prediction and the model's average prediction.

Three SHAP plots are generated:

- **Summary plot** (beeswarm). Each dot is one customer-feature combination. Position on the x-axis shows the feature's SHAP value (negative = pushing toward "active", positive = pushing toward "churn"). Color shows the feature value (red high, blue low). This gives global feature importance with direction.
- **Bar plot** - Mean absolute SHAP value per feature, ranked. Simpler version of the summary plot, useful for slide decks.
- **Waterfall plot** - Explains one specific high-probability prediction. Starts from the model's average prediction, then adds each feature's SHAP contribution in order of magnitude until reaching the final prediction. This is the plot a customer success manager would look at when reviewing a flagged account.

The waterfall is the most actionable output of the project. It tells the customer success team not just "this customer is high risk" but "this customer is high risk because their login frequency dropped 60%, they have three unresolved tickets, and they're on a month-to-month contract." Those are specific facts that suggest specific interventions.

## MLflow Tracking

Every training run is logged to MLflow under the experiment `saas-churn-prediction`. Each run captures:

- All model hyperparameters
- Test set metrics (accuracy, precision, recall, F1, ROC-AUC)
- Cross-validation mean ROC-AUC
- The serialized model

By default MLflow stores everything under `mlruns/` in the project root. To browse the runs:

```bash
mlflow ui
```

This starts a web UI at <http://localhost:5000> where you can compare runs, inspect parameters, and download model artifacts.

For a production deployment, MLflow's model registry would become a more significant part of the workflow - promoting models from staging to production, rolling back when needed. This project doesn't exercise that machinery but is structured so it could.

## Persisted Artifacts

The training pipeline writes three files to `models/`:

| File | Contents |
|---|---|
| `best_model.joblib` | The serialized winning model |
| `scaler.joblib` | The fitted StandardScaler |
| `model_metadata.joblib` | Threshold, metrics, feature names, revenue impact |

The metadata file is the contract between training and serving. Anything that consumes the model (API, dashboard, batch scoring) reads metadata first to know how the model should be used. This means I can change which model wins (logistic regression beating XGBoost on a particular run, say) without changing any downstream code - the metadata tells everyone what to do.

## Running the Training Pipeline

```bash
make train      # Trains all three models and selects the best
make evaluate   # Generates evaluation plots and SHAP explanations
make predict    # Runs batch scoring against the full dataset
```

Or as a single command:

```bash
python -m src.models.train && python -m src.models.evaluate && python -m src.models.predict
```

A full run takes about two minutes on a basic laptop, most of which is XGBoost training and SHAP computation.

## What I'd Improve Next

If I were taking this further:

- **Proper hyperparameter search.** Optuna with the MLflow tracking already in place would be a few hours of work and would likely squeeze a few more points of F1 out of the winning model.
- **Per-prediction SHAP in the API.** Right now the API returns a feature-magnitude approximation for the top risk factors. Real SHAP values would cost about 50ms per request - acceptable for most retention workflows, which aren't latency-sensitive.
- **Calibration analysis.** ROC-AUC tells you whether the model ranks customers correctly. Calibration tells you whether the predicted probabilities are trustworthy in absolute terms. Worth checking with a reliability diagram before using the probabilities for anything more than ranking.
- **Drift detection.** Compare the distribution of incoming features to the training distribution on a schedule, and alert when divergence exceeds a threshold.
