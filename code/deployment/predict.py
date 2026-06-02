"""Batch inference: takes a raw CSV with the same columns as training and emits
a CSV with fraud probabilities + hard predictions.

Usage:
    python code/deployment/predict.py --input path/to/input.csv --output path/to/preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import ID_COLS, MODELS, REPORTS, TARGET_COL

try:
    from deployment.monitor import check_drift, log_predictions
except ImportError:
    from monitor import check_drift, log_predictions

LOGGER = get_logger("predict")


def _load_artifacts() -> tuple:
    schema_path = MODELS / "feature_schema.json"
    encoder_path = MODELS / "encoder.pkl"
    model_path = MODELS / "flaml_automl_model.pkl"
    for p in (schema_path, encoder_path, model_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    with schema_path.open("r", encoding="utf-8") as fp:
        schema = json.load(fp)
    encoder = joblib.load(encoder_path)
    model = joblib.load(model_path)
    LOGGER.info("Loaded encoder, model, and schema from %s", MODELS)
    return schema, encoder, model


def _validate_input(df: pd.DataFrame, schema: dict) -> None:
    """Raise ValueError if df is missing required columns or is empty."""
    if df.empty:
        raise ValueError("Input CSV is empty (0 rows).")

    numeric_required = schema["numeric"]
    categorical_required = schema["categorical"]
    required_cols = set(numeric_required + categorical_required)
    present_cols = set(df.columns)
    missing = required_cols - present_cols

    if missing:
        raise ValueError(
            f"Input CSV is missing {len(missing)} required column(s): "
            f"{sorted(missing)}. "
            f"Present columns: {sorted(present_cols)}."
        )

    for col in numeric_required:
        null_count = df[col].isna().sum()
        if null_count:
            LOGGER.warning("Column '%s' has %d null value(s); will be passed through as-is.", col, null_count)


def _engineer(df: pd.DataFrame, schema: dict, encoder) -> pd.DataFrame:
    numeric = schema["numeric"]
    categorical = schema["categorical"]
    expected = schema["feature_columns"]

    if categorical:
        encoded = encoder.transform(df[categorical])
        encoded_cols = encoder.get_feature_names_out(categorical).tolist()
        X_cat = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        X = pd.concat([df[numeric].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = df[numeric].copy()

    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected]


def predict(input_path: Path, output_path: Path) -> Path:
    schema, encoder, model = _load_artifacts()

    LOGGER.info("Reading input %s", input_path)
    df = pd.read_csv(input_path)
    LOGGER.info("Input shape: %s", df.shape)

    id_frame = df[[c for c in ID_COLS if c in df.columns]].copy()
    feature_input = df.drop(columns=[c for c in (ID_COLS + [TARGET_COL]) if c in df.columns])

    _validate_input(feature_input, schema)
    X = _engineer(feature_input, schema, encoder)
    LOGGER.info("Engineered shape: %s", X.shape)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    out = id_frame.copy()
    out["fraud_probability"] = proba
    out["predicted_fraud"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d predictions to %s", len(out), output_path)

    batch_id = input_path.stem
    log_predictions(out, batch_id)

    drift_results = check_drift(feature_input, schema)
    if drift_results:
        drifted_features = [f for f, v in drift_results.items() if v["drifted"]]
        if drifted_features:
            LOGGER.warning(
                "Drift detected in %d feature(s): %s",
                len(drifted_features),
                drifted_features,
            )
        else:
            LOGGER.info("No drift detected in %d tested feature(s)", len(drift_results))

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fraud prediction")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "predictions.csv",
        help="Where to write the predictions CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("=== Prediction stage ===")
    predict(args.input, args.output)
    LOGGER.info("Prediction complete")


if __name__ == "__main__":
    main()
