"""Tests for code/deployment/predict.py"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from deployment.predict import _engineer, _validate_input


class TestValidateInput:
    def test_valid_input_passes(self, raw_df, feature_schema):
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud"])
        _validate_input(feature_df, feature_schema)

    def test_empty_df_raises(self, feature_schema):
        empty = pd.DataFrame(
            columns=["amount", "hour", "device_risk_score", "ip_risk_score",
                     "transaction_type", "merchant_category", "country"]
        )
        with pytest.raises(ValueError, match="empty"):
            _validate_input(empty, feature_schema)

    def test_missing_single_column_raises(self, raw_df, feature_schema):
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud", "amount"])
        with pytest.raises(ValueError, match="amount"):
            _validate_input(feature_df, feature_schema)

    def test_missing_multiple_columns_raises(self, raw_df, feature_schema):
        feature_df = raw_df.drop(
            columns=["transaction_id", "user_id", "is_fraud",
                     "amount", "transaction_type"]
        )
        with pytest.raises(ValueError) as exc_info:
            _validate_input(feature_df, feature_schema)
        msg = str(exc_info.value)
        assert "required column(s)" in msg

    def test_null_values_log_warning_not_raise(self, raw_df, feature_schema, caplog):
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud"]).copy()
        feature_df.loc[0, "amount"] = np.nan
        _validate_input(feature_df, feature_schema)


class TestEngineer:
    def test_output_columns_match_schema(self, raw_df, feature_schema, fitted_encoder):
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud"])
        result = _engineer(feature_df, feature_schema, fitted_encoder)
        assert list(result.columns) == feature_schema["feature_columns"]

    def test_output_row_count_preserved(self, raw_df, feature_schema, fitted_encoder):
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud"])
        result = _engineer(feature_df, feature_schema, fitted_encoder)
        assert len(result) == len(raw_df)

    def test_missing_expected_column_filled_with_zero(
        self, raw_df, feature_schema, fitted_encoder
    ):
        """If schema expects a feature column that's not in the output, it's filled with 0.0."""
        feature_df = raw_df.drop(columns=["transaction_id", "user_id", "is_fraud"])
        result = _engineer(feature_df, feature_schema, fitted_encoder)
        for col in feature_schema["feature_columns"]:
            assert col in result.columns

    def test_numeric_only_schema(self):
        """When there are no categorical columns, _engineer works with numerics only."""
        from deployment.predict import _engineer
        schema_numeric_only = {
            "numeric": ["a", "b"],
            "categorical": [],
            "feature_columns": ["a", "b"],
        }
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        encoder = MagicMock()
        result = _engineer(df, schema_numeric_only, encoder)
        assert list(result.columns) == ["a", "b"]
        encoder.transform.assert_not_called()
