"""FLAML AutoML training, optimised for F1 on the fraud (positive) class.

We deliberately ignore accuracy because the dataset is imbalanced. Reported and
optimised metrics are precision, recall, F1, and PR-AUC.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flaml import AutoML
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import DATA_PROCESSED, ID_COLS, MODELS, RAW_CSV, REPORTS, TARGET_COL, load_config

LOGGER = get_logger("train")


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv")[TARGET_COL]
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")[TARGET_COL]
    LOGGER.info("Loaded splits: X_train=%s X_test=%s", X_train.shape, X_test.shape)

    # Validate data integrity
    assert len(X_train) == len(y_train), f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
    assert len(X_test) == len(y_test), f"X_test and y_test length mismatch: {len(X_test)} vs {len(y_test)}"
    assert X_train.shape[1] == X_test.shape[1], f"Feature count mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
    assert list(X_train.columns) == list(X_test.columns), "Column names don't match between train and test"

    # Check for duplicates across train/test
    X_combined = pd.concat([X_train.reset_index(drop=True), X_test.reset_index(drop=True)], ignore_index=True)
    n_duplicates = X_combined.duplicated(keep=False).sum()
    if n_duplicates > 0:
        LOGGER.warning("Found %d duplicate rows across train/test sets (potential data leakage!)", n_duplicates)
        # Find which rows are duplicated
        dup_mask = X_combined.duplicated(keep=False)
        dup_indices = X_combined[dup_mask].index.tolist()
        train_dup = sum(1 for idx in dup_indices if idx < len(X_train))
        test_dup = sum(1 for idx in dup_indices if idx >= len(X_train))
        LOGGER.warning(f"  Train duplicates: {train_dup}, Test duplicates: {test_dup}")

    LOGGER.info("Data validation passed | train targets: %s | test targets: %s",
                dict(y_train.value_counts()), dict(y_test.value_counts()))

    return X_train, X_test, y_train, y_test


def _plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["legit", "fraud"],
        yticklabels=["legit", "fraud"],
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (holdout)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve (fraud class)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_reference_stats() -> None:
    """Reconstruct pre-SMOTE X_train and save reference distribution stats for drift detection."""
    cfg = load_config()
    random_state = cfg["preprocessing"]["random_state"]
    test_size = cfg["preprocessing"]["test_size"]

    schema_path = MODELS / "feature_schema.json"
    if not schema_path.exists():
        LOGGER.warning("feature_schema.json not found; skipping reference stats")
        return

    with schema_path.open("r", encoding="utf-8") as fp:
        schema = json.load(fp)

    raw_df = pd.read_csv(RAW_CSV)
    raw_df = raw_df.drop_duplicates().reset_index(drop=True)

    numeric = schema["numeric"]
    categorical = schema["categorical"]
    y = raw_df[TARGET_COL].astype(int)
    X_raw = raw_df.drop(columns=ID_COLS + [TARGET_COL])

    X_pre_smote, _, _, _ = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )

    ref_stats = {}
    for col in numeric:
        if col in X_pre_smote.columns:
            s = X_pre_smote[col].dropna()
            ref_stats[col] = {
                "type": "numeric",
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p05": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
            }

    for col in categorical:
        if col in X_pre_smote.columns:
            proportions = X_pre_smote[col].value_counts(normalize=True).to_dict()
            ref_stats[col] = {
                "type": "categorical",
                "proportions": {str(k): float(v) for k, v in proportions.items()},
            }

    out_path = MODELS / "reference_stats.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(ref_stats, fp, indent=2)
    LOGGER.info("Saved reference stats to %s", out_path)


def train() -> dict:
    cfg = load_config()
    TIME_BUDGET_SECONDS = cfg["training"]["time_budget_seconds"]
    FLAML_METRIC = cfg["training"]["metric"]
    TASK = cfg["training"]["task"]
    ESTIMATORS = cfg["training"]["estimators"]
    N_SPLITS = cfg["training"]["n_splits"]
    SEED = cfg["training"]["seed"]

    X_train, X_test, y_train, y_test = _load_splits()

    automl = AutoML()
    settings = {
        "time_budget": TIME_BUDGET_SECONDS,
        "metric": FLAML_METRIC,
        "task": TASK,
        "estimator_list": ESTIMATORS,
        "eval_method": "cv",
        "n_splits": N_SPLITS,
        "seed": SEED,
        "log_file_name": str(REPORTS / "flaml_search.log"),
        "verbose": 1,
    }
    LOGGER.info("Starting FLAML AutoML with settings=%s", settings)
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    LOGGER.info("Best estimator: %s", automl.best_estimator)
    LOGGER.info("Best CV loss (1 - %s): %.5f", FLAML_METRIC, automl.best_loss)
    LOGGER.info("Best config: %s", automl.best_config)

    y_pred = automl.predict(X_test)
    if hasattr(automl, "predict_proba"):
        y_prob = automl.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "best_estimator": automl.best_estimator,
        "best_cv_loss": float(automl.best_loss),
        "best_config": automl.best_config,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    LOGGER.info(
        "Holdout metrics | precision=%.4f recall=%.4f f1=%.4f PR-AUC=%.4f ROC-AUC=%.4f",
        precision,
        recall,
        f1,
        pr_auc,
        roc_auc,
    )

    # Warn if metrics are suspiciously perfect (potential data leakage or dataset too simple)
    if precision == 1.0 and recall == 1.0 and f1 == 1.0:
        LOGGER.warning(
            "WARNING: Perfect metrics detected (precision=1.0, recall=1.0, f1=1.0). "
            "This could indicate: (1) data leakage between train/test, (2) synthetic dataset with perfect separation, "
            "or (3) model memorizing data. Recommend: "
            "(a) verify train/test split isolation, (b) check feature correlations with target, "
            "(c) validate on hold-out set from different time period or data distribution."
        )

    report_txt = classification_report(y_test, y_pred, target_names=["legit", "fraud"], digits=4)
    LOGGER.info("Classification report:\n%s", report_txt)

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion(cm, REPORTS / "confusion_matrix.png")
    _plot_pr_curve(y_test, y_prob, REPORTS / "precision_recall_curve.png")

    with (REPORTS / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, default=str)
    with (REPORTS / "classification_report.txt").open("w", encoding="utf-8") as fp:
        fp.write(report_txt)
    pd.DataFrame(cm, index=["legit", "fraud"], columns=["legit", "fraud"]).to_csv(
        REPORTS / "confusion_matrix.csv"
    )

    model_path = MODELS / "flaml_automl_model.pkl"
    joblib.dump(automl, model_path)
    LOGGER.info("Persisted FLAML model -> %s", model_path)

    _save_reference_stats()

    return metrics


def main() -> None:
    LOGGER.info("=== Modeling stage (FLAML AutoML) ===")
    train()
    LOGGER.info("Training complete")


if __name__ == "__main__":
    main()
