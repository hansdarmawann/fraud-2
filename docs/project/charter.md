# Project Charter — Synthetic Fraud Detection

## Business problem
Detect fraudulent transactions from a synthetic transaction dataset. The cost of
missing fraud (false negative) is materially higher than the cost of flagging a
legitimate transaction (false positive), so the model must be tuned to keep
**recall high** while preserving a usable **precision** — i.e. we optimise for
**F1 on the positive (fraud) class** and explicitly de-prioritise accuracy.

## Scope
- In scope: binary classification of `is_fraud` from tabular features.
- Out of scope: real-time scoring infrastructure, drift monitoring, online learning.

## Success metrics
| Metric                  | Target              |
|-------------------------|---------------------|
| Recall (fraud class)    | maximise            |
| Precision (fraud class) | keep usable (>0.20) |
| F1 (fraud class)        | primary optimisation target |
| PR-AUC                  | reported, threshold-free check |
| Accuracy                | **not** a target — meaningless on imbalanced data |

## Plan / methodology
Microsoft **TDSP** lifecycle:
1. Business understanding — this charter.
2. Data acquisition & understanding — EDA (univariate + multivariate).
3. Data preparation — cleaning, encoding, train/test split, SMOTE oversampling
   on the training partition only.
4. Modeling — FLAML AutoML, time-budgeted search optimising `f1`.
5. Deployment — batch scoring script (`code/deployment/predict.py`).

## Data
- Source: `data/raw/synthetic_fraud_dataset.csv` (10,000 rows, 10 columns).
- Target: `is_fraud` (0/1, imbalanced).
- Identifiers (dropped from features): `transaction_id`, `user_id`.
