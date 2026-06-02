"""Debug script to check for data leakage between train and test sets."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.paths import DATA_PROCESSED, RAW_CSV

LOGGER = get_logger("debug_leakage")

def check_leakage():
    """Check for train/test overlap and data integrity."""

    # Load data
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")

    LOGGER.info("=== Data Shapes ===")
    LOGGER.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    LOGGER.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Check for duplicates within each set
    LOGGER.info("\n=== Duplicates ===")
    dup_train = X_train.duplicated().sum()
    dup_test = X_test.duplicated().sum()
    LOGGER.info(f"Duplicates in X_train: {dup_train}")
    LOGGER.info(f"Duplicates in X_test: {dup_test}")

    # Check for test rows in training data (sample check due to size)
    LOGGER.info("\n=== Checking for test rows in training data ===")
    test_in_train_count = 0
    for i in range(min(100, len(X_test))):  # Check first 100 test rows
        test_row = X_test.iloc[i].values
        match = (X_train.values == test_row).all(axis=1)
        if match.any():
            test_in_train_count += 1
            LOGGER.warning(f"Test row {i} found in training data!")

    LOGGER.info(f"Test rows found in training data: {test_in_train_count} / 100 checked")

    # Check feature distributions
    LOGGER.info("\n=== Feature Statistics ===")
    LOGGER.info("X_train numeric cols (sample):")
    LOGGER.info(X_train[["amount", "hour", "device_risk_score", "ip_risk_score"]].describe().T[["min", "mean", "max"]])

    LOGGER.info("\nX_test numeric cols (sample):")
    LOGGER.info(X_test[["amount", "hour", "device_risk_score", "ip_risk_score"]].describe().T[["min", "mean", "max"]])

    # Check target distribution
    LOGGER.info("\n=== Target Distribution ===")
    LOGGER.info(f"y_train value counts:\n{y_train['is_fraud'].value_counts()}")
    LOGGER.info(f"y_test value counts:\n{y_test['is_fraud'].value_counts()}")

    # Check for exact duplicates between datasets
    LOGGER.info("\n=== Exact duplicate rows across train/test ===")
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True).reset_index(drop=True)

    duplicates = X_combined.duplicated(keep=False)
    dup_rows = X_combined[duplicates].index.tolist()

    if dup_rows:
        LOGGER.warning(f"Found {len(dup_rows)} duplicate rows across train/test")
        for idx in dup_rows[:5]:
            y_val = y_combined.iloc[idx]
            dataset = "TRAIN" if idx < len(X_train) else "TEST"
            LOGGER.warning(f"  Row {idx} ({dataset}): y={y_val}")
    else:
        LOGGER.info("No exact duplicate rows across train/test ✓")

    # Load raw data for comparison
    LOGGER.info("\n=== Checking raw data size vs processed ===")
    raw_df = pd.read_csv(RAW_CSV)
    LOGGER.info(f"Raw dataset shape: {raw_df.shape}")
    LOGGER.info(f"Train + Test combined: ({len(X_train)}, {X_train.shape[1]}) + ({len(X_test)}, {X_test.shape[1]})")

    # Check if any row appears in both train and test with different labels
    LOGGER.info("\n=== Checking for same row with different labels ===")
    X_train_test = pd.merge(X_train.reset_index(drop=True), X_test.reset_index(drop=True),
                            how='inner', on=list(X_train.columns))
    if len(X_train_test) > 0:
        LOGGER.warning(f"Found {len(X_train_test)} identical rows in both train and test!")
    else:
        LOGGER.info("No identical rows across train/test ✓")

if __name__ == "__main__":
    check_leakage()
