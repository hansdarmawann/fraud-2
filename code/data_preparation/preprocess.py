"""Preprocessing: clean, encode, split, then oversample the training side only.

Critical design choices:
  * SMOTE is applied AFTER the train/test split, so the holdout reflects the
    true (imbalanced) population.
  * Categorical features are one-hot encoded; encoder + feature columns are
    persisted so inference can reproduce the same column space.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import DATA_PROCESSED, ID_COLS, MODELS, RAW_CSV, TARGET_COL, load_config

LOGGER = get_logger("preprocess")


def _split_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    drop = set(ID_COLS + [TARGET_COL])
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    categorical = [
        c for c in df.select_dtypes(include=["object", "category"]).columns if c not in drop
    ]
    LOGGER.info("Numeric features (%d): %s", len(numeric), numeric)
    LOGGER.info("Categorical features (%d): %s", len(categorical), categorical)
    return numeric, categorical


def _build_encoder() -> OneHotEncoder:
    # sklearn renamed `sparse` -> `sparse_output` in 1.2; support both.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess() -> dict:
    cfg = load_config()
    RANDOM_STATE = cfg["preprocessing"]["random_state"]
    TEST_SIZE = cfg["preprocessing"]["test_size"]

    LOGGER.info("Loading raw data from %s", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    LOGGER.info("Raw shape: %s", df.shape)

    df = df.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Post-deduplication shape: %s", df.shape)

    missing = df.isna().sum().sum()
    if missing:
        LOGGER.warning("Found %d missing values; filling numerics with median, cats with mode", missing)
        for c in df.columns:
            if df[c].isna().any():
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])

    numeric, categorical = _split_features(df)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=ID_COLS + [TARGET_COL])

    encoder = _build_encoder()
    if categorical:
        encoded = encoder.fit_transform(X[categorical])
        encoded_cols = encoder.get_feature_names_out(categorical).tolist()
        X_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
        X_final = pd.concat([X[numeric].reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    else:
        X_final = X[numeric].copy()
        encoded_cols = []

    LOGGER.info("Engineered feature matrix shape: %s", X_final.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    LOGGER.info(
        "Pre-SMOTE | train=%s test=%s | train positives=%d (%.2f%%) | test positives=%d (%.2f%%)",
        X_train.shape,
        X_test.shape,
        int(y_train.sum()),
        100 * y_train.mean(),
        int(y_test.sum()),
        100 * y_test.mean(),
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    LOGGER.info(
        "Post-SMOTE | train=%s | positives=%d (%.2f%%)",
        X_train_res.shape,
        int(y_train_res.sum()),
        100 * y_train_res.mean(),
    )

    # Ensure SMOTE output is a DataFrame with correct columns
    if not isinstance(X_train_res, pd.DataFrame):
        X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    else:
        # SMOTE might lose column names, so explicitly set them
        X_train_res.columns = X_train.columns

    # Ensure y_train_res is a Series
    if not isinstance(y_train_res, pd.Series):
        y_train_res = pd.Series(y_train_res, name=TARGET_COL)

    LOGGER.info("Post-SMOTE data types | X_train_res: %s, y_train_res: %s",
                type(X_train_res).__name__, type(y_train_res).__name__)

    # Validate data alignment
    assert len(X_train_res) == len(y_train_res), \
        f"Train set misalignment: {len(X_train_res)} features vs {len(y_train_res)} labels"
    assert X_train_res.shape[1] == X_test.shape[1], \
        f"Feature column mismatch: train={X_train_res.shape[1]}, test={X_test.shape[1]}"
    assert list(X_train_res.columns) == list(X_test.columns), \
        f"Column names don't match: train columns={list(X_train_res.columns[:3])}..., test columns={list(X_test.columns[:3])}..."

    # Check for NaN values
    nan_train = X_train_res.isna().sum().sum() + y_train_res.isna().sum()
    nan_test = X_test.isna().sum().sum() + y_test.isna().sum()
    if nan_train > 0:
        LOGGER.warning(f"Found {nan_train} NaN values in training set!")
    if nan_test > 0:
        LOGGER.warning(f"Found {nan_test} NaN values in test set!")

    # Log feature ranges for debugging
    LOGGER.info("Feature ranges in X_train_res (first 3): min=%s, max=%s",
                {col: f"{X_train_res[col].min():.4f}" for col in list(X_train_res.columns)[:3]},
                {col: f"{X_train_res[col].max():.4f}" for col in list(X_train_res.columns)[:3]})

    X_train_res.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train_res.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    pd.Series(y_test, name=TARGET_COL).to_csv(DATA_PROCESSED / "y_test.csv", index=False)
    LOGGER.info("Wrote train/test splits to %s", DATA_PROCESSED)

    # Verify files were written correctly
    X_train_check = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y_train_check = pd.read_csv(DATA_PROCESSED / "y_train.csv")
    assert X_train_check.shape == X_train_res.shape, \
        f"X_train write verification failed: {X_train_res.shape} -> {X_train_check.shape}"
    assert len(y_train_check) == len(y_train_res), \
        f"y_train write verification failed: {len(y_train_res)} -> {len(y_train_check)}"
    LOGGER.info("File write verification passed")

    joblib.dump(encoder, MODELS / "encoder.pkl")
    schema = {
        "numeric": numeric,
        "categorical": categorical,
        "encoded_categorical": encoded_cols,
        "feature_columns": X_final.columns.tolist(),
        "target": TARGET_COL,
        "id_columns": ID_COLS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with (MODELS / "feature_schema.json").open("w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2)
    LOGGER.info("Persisted encoder + feature schema under %s", MODELS)

    return {
        "X_train": X_train_res,
        "X_test": X_test,
        "y_train": y_train_res,
        "y_test": y_test,
        "schema": schema,
    }


def main() -> None:
    LOGGER.info("=== Preprocessing stage ===")
    preprocess()
    LOGGER.info("Preprocessing complete")


if __name__ == "__main__":
    main()
