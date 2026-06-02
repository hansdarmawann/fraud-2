"""Model monitoring: drift detection and prediction logging."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import MODELS, REPORTS

LOGGER = get_logger("monitor")


def _sample_from_stats(ref: dict, n: int = 1000) -> np.ndarray:
    """Reconstruct a reference sample from stored percentile quantiles.

    Uses inverse-CDF sampling via linear interpolation of stored quantiles.
    """
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    values = [ref["p05"], ref["p25"], ref["p50"], ref["p75"], ref["p95"]]
    u = np.random.default_rng(42).uniform(0.05, 0.95, size=n)
    sample = np.interp(u, quantiles, values)
    return sample


def check_drift(current_df: pd.DataFrame, schema: dict) -> dict:
    """Check for data drift in numeric and categorical features.

    Compares current batch against stored reference statistics using KS test
    (numeric) and chi-square test (categorical). Features with p < 0.05 are
    flagged as drifted.

    Returns dict {feature: {statistic, pvalue, drifted}}, or {} if reference
    stats not found.
    """
    ref_path = MODELS / "reference_stats.json"
    if not ref_path.exists():
        LOGGER.warning("reference_stats.json not found; skipping drift check")
        return {}

    with ref_path.open("r", encoding="utf-8") as fp:
        ref_stats = json.load(fp)

    results = {}
    numeric_cols = schema.get("numeric", [])
    categorical_cols = schema.get("categorical", [])

    for col in numeric_cols:
        if col not in ref_stats or col not in current_df.columns:
            continue
        ref = ref_stats[col]
        current_vals = current_df[col].dropna().values
        if len(current_vals) < 2:
            continue
        ref_sample = _sample_from_stats(ref)
        stat, pvalue = stats.ks_2samp(ref_sample, current_vals)
        drifted = pvalue < 0.05
        results[col] = {"statistic": float(stat), "pvalue": float(pvalue), "drifted": drifted}
        if drifted:
            LOGGER.warning("DRIFT DETECTED in '%s': KS stat=%.4f p=%.4f", col, stat, pvalue)

    for col in categorical_cols:
        if col not in ref_stats or col not in current_df.columns:
            continue
        ref = ref_stats[col]
        ref_proportions = ref["proportions"]
        observed_counts = current_df[col].value_counts()
        all_cats = sorted(ref_proportions.keys())
        observed_freq = [observed_counts.get(cat, 0) for cat in all_cats]
        expected_freq = [ref_proportions.get(cat, 0) * len(current_df) for cat in all_cats]
        if sum(expected_freq) == 0 or min(expected_freq) < 5:
            LOGGER.debug("Skipping chi-square for '%s': low expected counts", col)
            continue
        stat, pvalue = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
        drifted = pvalue < 0.05
        results[col] = {"statistic": float(stat), "pvalue": float(pvalue), "drifted": drifted}
        if drifted:
            LOGGER.warning("DRIFT DETECTED in '%s': chi2 stat=%.4f p=%.4f", col, stat, pvalue)

    return results


def log_predictions(predictions_df: pd.DataFrame, batch_id: str) -> None:
    """Log prediction batch statistics to a CSV for trend analysis.

    Appends: batch_id, timestamp, n_predictions, fraud_rate.
    """
    log_path = REPORTS / "prediction_log.csv"
    fraud_rate = (
        float(predictions_df["predicted_fraud"].mean())
        if "predicted_fraud" in predictions_df.columns
        else float("nan")
    )
    row = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_predictions": len(predictions_df),
        "fraud_rate": round(fraud_rate, 6),
    }
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    LOGGER.info("Logged batch '%s': n=%d fraud_rate=%.4f", batch_id, len(predictions_df), fraud_rate)
