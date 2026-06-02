"""Shared test fixtures for fraud-1 project."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


@pytest.fixture()
def raw_df():
    """Minimal synthetic DataFrame matching the real dataset schema."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "transaction_id": range(n),
        "user_id": range(n),
        "amount": rng.uniform(10, 500, n),
        "transaction_type": rng.choice(["POS", "ATM", "Online", "QR"], n),
        "merchant_category": rng.choice(
            ["Food", "Grocery", "Electronics", "Clothing", "Travel"], n
        ),
        "country": rng.choice(["US", "UK", "DE", "FR", "NG", "TR"], n),
        "hour": rng.integers(0, 24, n),
        "device_risk_score": rng.uniform(0, 1, n),
        "ip_risk_score": rng.uniform(0, 1, n),
        "is_fraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture()
def feature_schema():
    """Standard feature schema matching the training pipeline output."""
    return {
        "numeric": ["amount", "hour", "device_risk_score", "ip_risk_score"],
        "categorical": ["transaction_type", "merchant_category", "country"],
        "encoded_categorical": [
            "transaction_type_ATM", "transaction_type_Online",
            "transaction_type_POS", "transaction_type_QR",
            "merchant_category_Clothing", "merchant_category_Electronics",
            "merchant_category_Food", "merchant_category_Grocery",
            "merchant_category_Travel",
            "country_DE", "country_FR", "country_NG",
            "country_TR", "country_UK", "country_US",
        ],
        "feature_columns": [
            "amount", "hour", "device_risk_score", "ip_risk_score",
            "transaction_type_ATM", "transaction_type_Online",
            "transaction_type_POS", "transaction_type_QR",
            "merchant_category_Clothing", "merchant_category_Electronics",
            "merchant_category_Food", "merchant_category_Grocery",
            "merchant_category_Travel",
            "country_DE", "country_FR", "country_NG",
            "country_TR", "country_UK", "country_US",
        ],
        "target": "is_fraud",
        "id_columns": ["transaction_id", "user_id"],
        "test_size": 0.2,
        "random_state": 42,
    }


@pytest.fixture()
def fitted_encoder(raw_df, feature_schema):
    """A OneHotEncoder fitted on the synthetic raw_df."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(raw_df[feature_schema["categorical"]])
    return enc
