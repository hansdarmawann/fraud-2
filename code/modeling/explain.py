"""SHAP-based model explainability. Generates feature importance plots and statistics.

Usage:
    python -c "import sys; sys.path.insert(0, 'code'); from modeling.explain import explain; explain()"
Or as a pipeline stage:
    python main.py --stages train explain
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import DATA_PROCESSED, FIGURES, MODELS, REPORTS

LOGGER = get_logger("explain")


def explain() -> dict:
    """Generate SHAP explanations for the trained model."""
    LOGGER.info("Loading artifacts for SHAP analysis")
    model_path = MODELS / "flaml_automl_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    automl = joblib.load(model_path)
    estimator = automl.model.estimator
    LOGGER.info("Loaded estimator: %s", type(estimator).__name__)

    X_test_path = DATA_PROCESSED / "X_test.csv"
    X_test = pd.read_csv(X_test_path)
    LOGGER.info("Loaded X_test: shape=%s", X_test.shape)

    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    LOGGER.info("Using background sample: shape=%s", X_sample.shape)

    LOGGER.info("Creating SHAP explainer (auto-detecting model type)")
    explainer = shap.Explainer(estimator, X_sample)

    LOGGER.info("Computing SHAP values (this may take a minute)")
    shap_values = explainer(X_test)

    sv = shap_values if shap_values.values.ndim == 2 else shap_values[:, :, 1]
    LOGGER.info("SHAP values shape: %s", sv.values.shape)

    LOGGER.info("Generating beeswarm plot")
    shap.plots.beeswarm(sv, max_display=20, show=False)
    plt.savefig(FIGURES / "shap_summary_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved beeswarm plot -> %s", FIGURES / "shap_summary_beeswarm.png")

    LOGGER.info("Generating bar plot")
    shap.plots.bar(sv, max_display=20, show=False)
    plt.savefig(FIGURES / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved bar plot -> %s", FIGURES / "shap_bar.png")

    mean_abs_shap = np.abs(sv.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    output_path = REPORTS / "shap_mean_abs.csv"
    feature_importance.to_csv(output_path, index=False)
    LOGGER.info("Saved feature importance CSV -> %s", output_path)

    top_5 = feature_importance.head(5)
    LOGGER.info("Top 5 features by mean |SHAP|:\n%s", top_5.to_string())

    return {
        "n_features": len(feature_importance),
        "top_feature": feature_importance.iloc[0]["feature"],
        "top_feature_shap": float(feature_importance.iloc[0]["mean_abs_shap"]),
    }


def main() -> None:
    LOGGER.info("=== SHAP Explanation stage ===")
    explain()
    LOGGER.info("Explanation complete")


if __name__ == "__main__":
    main()
