# Model Report

## Method
**FLAML AutoML** by Microsoft Research. Time-budgeted (default: 120 seconds)
hyperparameter and learner search across:

- LightGBM
- XGBoost
- Random Forest
- Extra Trees
- L1 Logistic Regression

Search objective: **F1 score** (FLAML internally minimises `1 - F1`) with 5-fold
cross-validation on the SMOTE-oversampled training partition.

## Why not accuracy
The target is imbalanced. A trivial "always predict legit" classifier would score
high accuracy while having zero recall on fraud — useless. We report and optimise
precision / recall / F1 / PR-AUC on the fraud class; accuracy is intentionally
excluded from the optimisation objective.

## Evaluation
Evaluation runs on the **untouched** stratified holdout (20% of original data,
no SMOTE applied). Generated artifacts:

- `outputs/reports/metrics.json` — best estimator + holdout metrics.
- `outputs/reports/classification_report.txt` — per-class precision/recall/F1.
- `outputs/reports/confusion_matrix.csv` / `.png` — confusion matrix.
- `outputs/reports/precision_recall_curve.png` — PR curve with average precision.
- `outputs/models/flaml_automl_model.pkl` — serialised AutoML object.
- `outputs/models/encoder.pkl` — OneHotEncoder for categorical features.
- `outputs/models/feature_schema.json` — feature column order required at inference.

## Inference
```bash
python code/deployment/predict.py --input path/to/new_data.csv \
    --output outputs/reports/predictions.csv
```
The script reapplies the persisted encoder, aligns to the training feature
schema, and writes `fraud_probability` + `predicted_fraud` per row.
