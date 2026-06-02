"""Tests for code/data_preparation/preprocess.py"""

import numpy as np
import pandas as pd
import pytest

from data_preparation.preprocess import _build_encoder, _split_features


class TestSplitFeatures:
    def test_numeric_and_categorical_split(self, raw_df):
        numeric, categorical = _split_features(raw_df)
        assert set(numeric) == {"amount", "hour", "device_risk_score", "ip_risk_score"}
        assert set(categorical) == {"transaction_type", "merchant_category", "country"}

    def test_id_and_target_cols_excluded(self, raw_df):
        numeric, categorical = _split_features(raw_df)
        all_cols = numeric + categorical
        assert "transaction_id" not in all_cols
        assert "user_id" not in all_cols
        assert "is_fraud" not in all_cols

    def test_all_numeric_df(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "is_fraud": [0, 1]})
        numeric, categorical = _split_features(df)
        assert "a" in numeric and "b" in numeric
        assert categorical == []

    def test_all_categorical_df(self):
        df = pd.DataFrame(
            {
                "cat_a": ["x", "y"],
                "cat_b": ["p", "q"],
                "is_fraud": [0, 1],
            }
        )
        numeric, categorical = _split_features(df)
        assert numeric == []
        assert set(categorical) == {"cat_a", "cat_b"}


class TestBuildEncoder:
    def test_returns_encoder(self):
        enc = _build_encoder()
        from sklearn.preprocessing import OneHotEncoder
        assert isinstance(enc, OneHotEncoder)

    def test_handle_unknown_ignore(self):
        enc = _build_encoder()
        assert enc.handle_unknown == "ignore"

    def test_encoder_fits_and_transforms(self, raw_df, feature_schema):
        enc = _build_encoder()
        cats = feature_schema["categorical"]
        enc.fit(raw_df[cats])
        result = enc.transform(raw_df[cats])
        assert result.shape[0] == len(raw_df)
        assert result.shape[1] > 0

    def test_encoder_handles_unknown_category(self, raw_df, feature_schema):
        enc = _build_encoder()
        cats = feature_schema["categorical"]
        enc.fit(raw_df[cats])
        test_row = raw_df[cats].iloc[:1].copy()
        test_row["transaction_type"] = "UNKNOWN_TYPE"
        result = enc.transform(test_row)
        assert result.shape == (1, len(enc.get_feature_names_out(cats)))
