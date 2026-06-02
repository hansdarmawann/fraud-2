# Fraud Detection — FLAML AutoML (TDSP)

Binary fraud-classification project on `synthetic_fraud_dataset.csv`, organised
under Microsoft's [Team Data Science Process](https://learn.microsoft.com/en-us/azure/architecture/data-science-process/overview)
folder convention.

> **Metric focus:** precision, recall, F1 (fraud class). Accuracy is *not* an
> optimisation target — the dataset is imbalanced and accuracy is misleading.

## Project layout

```
fraud-1/
├── code/
│   ├── data_acquisition_understanding/eda.py    # univariate + multivariate EDA
│   ├── data_preparation/preprocess.py           # cleaning, encoding, split, SMOTE
│   ├── modeling/train.py                        # FLAML AutoML
│   ├── deployment/predict.py                    # batch inference
│   └── utils/{logger.py, paths.py}              # shared utilities
├── notebooks/                                   # interactive companions
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── data/{raw,processed,external}/
├── docs/{project,data_report,model}/            # TDSP documentation
├── outputs/{figures,models,reports,logs}/
├── environment.yml                              # conda env spec
├── requirements.txt                             # pip-only equivalent
└── main.py                                      # one-command pipeline
```

## Environment setup (conda)

```bash
conda env create -f environment.yml
conda activate fraud-ml
```

If you'd rather use plain pip:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python main.py
```

This sequentially executes:

1. **EDA** — writes plots to `outputs/figures/{univariate,multivariate}/` and
   summary tables to `outputs/reports/`.
2. **Preprocessing** — cleans, one-hot encodes, stratified train/test split,
   then applies **SMOTE** on the training partition only. Persists splits,
   encoder, and feature schema.
3. **Modeling** — FLAML AutoML (LightGBM / XGBoost / RF / ExtraTrees / L1-LR),
   120-second budget, optimising F1 with 5-fold CV. Evaluates on the untouched
   holdout and writes metrics + confusion matrix + PR curve.

You can also run a subset:

```bash
python main.py --stages eda
python main.py --stages preprocess train
```

Individual stages are runnable directly too:

```bash
python code/data_acquisition_understanding/eda.py
python code/data_preparation/preprocess.py
python code/modeling/train.py
```

## Interactive notebooks

The notebooks under `notebooks/` import the same modules used by `main.py`, so
they cannot drift from the pipeline:

```bash
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_preprocessing.ipynb
jupyter lab notebooks/03_modeling.ipynb
```

## Inference on new data

```bash
python code/deployment/predict.py --input data/raw/new_batch.csv \
    --output outputs/reports/predictions.csv
```

The input CSV must contain the same source columns as the training data
(identifiers + features); the `is_fraud` column is optional.

## Logging

Every stage initialises a named logger (see `code/utils/logger.py`) that writes
both to stdout and to a timestamped file under `outputs/logs/`. Filenames look
like `train_20260526_141512.log`.

## Metrics produced

After `python main.py` finishes, look in `outputs/reports/`:

- `metrics.json` — precision, recall, F1, PR-AUC, ROC-AUC, best estimator
- `classification_report.txt` — sklearn classification report
- `confusion_matrix.{csv,png}`
- `precision_recall_curve.png`
- `flaml_search.log` — FLAML's internal search history
