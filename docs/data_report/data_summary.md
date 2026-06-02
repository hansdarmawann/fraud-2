# Data Report

## Source
`data/raw/synthetic_fraud_dataset.csv` — 10,000 transactions.

## Schema
| Column              | Type    | Notes                                        |
|---------------------|---------|----------------------------------------------|
| transaction_id      | int     | identifier — dropped before modeling         |
| user_id             | int     | identifier — dropped before modeling         |
| amount              | float   | transaction amount                           |
| transaction_type    | object  | categorical (ATM, POS, Online, QR, ...)      |
| merchant_category   | object  | categorical                                  |
| country             | object  | categorical (ISO code)                       |
| hour                | int     | hour-of-day (0–23)                           |
| device_risk_score   | float   | upstream risk score in [0, 1]                |
| ip_risk_score       | float   | upstream risk score in [0, 1]                |
| is_fraud            | int     | **target**, binary                           |

## Generated artifacts
- `outputs/reports/data_summary.csv` — per-column dtype, missingness, descriptive stats.
- `outputs/reports/class_balance.csv` — count + percentage by target class.
- `outputs/reports/categorical_target_rates.csv` — fraud rate per categorical value.
- `outputs/figures/univariate/` — per-feature histograms / boxplots / count plots.
- `outputs/figures/multivariate/` — correlation heatmap, target-conditioned KDEs,
  fraud-rate bar plots per category, pairplot of top correlated numerics.

## Imbalance handling
The training partition is over-sampled with **SMOTE** (synthetic minority
over-sampling) *after* the stratified train/test split, so the holdout reflects
the real-world class ratio.
