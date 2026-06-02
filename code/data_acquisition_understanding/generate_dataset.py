"""Generate a realistically hard synthetic fraud dataset.

Design
------
Stage 1 – generate contextual features (transaction type, country, hour, amount)
          and compute a base fraud probability from those alone.
Stage 2 – sample exactly 500 fraud transactions proportional to the base score,
          then generate device/IP risk scores *conditional on the fraud label*
          using separate Beta distributions per class.  This gives precise
          control over class-conditional overlap.

Target statistics
-----------------
* Fraud rate                : exactly 5 % (500 / 10 000)
* device_risk_score AUROC   : ~0.82  (alone)
* ip_risk_score AUROC       : ~0.79  (alone)
* Combined full-model AUROC : ~0.93–0.96
* Expected F1 (fraud class) : ~0.75–0.85

Key contrast with the old dataset
----------------------------------
Old: legit scores ∈ [0.00, 0.30], fraud scores ∈ [0.70, 1.00] – zero overlap.
New: both distributions overlap substantially in [0.10, 0.60];
     no threshold on a single feature achieves > 90 % accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import RAW_CSV

LOGGER = get_logger("generate_dataset")

N = 10_000
N_FRAUD = 500       # exactly 5 %
SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate(n: int = N, n_fraud: int = N_FRAUD, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── IDs (shuffled so ordering carries no class signal) ───────────────────
    transaction_id = rng.permutation(n)
    user_id = rng.integers(1, 2001, size=n)

    # ── Categorical features ─────────────────────────────────────────────────
    transaction_type = rng.choice(
        ["Online", "ATM", "POS", "QR"], size=n, p=[0.30, 0.18, 0.37, 0.15]
    )
    merchant_category = rng.choice(
        ["Grocery", "Food", "Clothing", "Electronics", "Travel"],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    country = rng.choice(
        ["US", "UK", "DE", "FR", "TR", "NG"],
        size=n, p=[0.40, 0.18, 0.15, 0.15, 0.07, 0.05],
    )

    # Hour: bimodal (morning commute 7-9, evening 17-20), light late-night tail
    hour_w = np.array([
        0.4, 0.3, 0.3, 0.3, 0.4, 0.7, 1.3, 2.0, 2.5, 2.3,
        2.1, 2.2, 2.4, 2.5, 2.3, 2.0, 2.3, 3.0, 3.5, 3.0,
        2.5, 2.0, 1.5, 0.8,
    ], dtype=float)
    hour = rng.choice(24, size=n, p=hour_w / hour_w.sum())

    # Amount: log-normal, median ~$66
    amount = np.clip(rng.lognormal(4.2, 1.4, n), 1.0, 50_000.0).round(2)

    # ── Stage 1: base fraud score from contextual features only ─────────────
    txn_enc = {"Online": 0.55, "ATM": 0.35, "QR": 0.10, "POS": -0.10}
    cat_enc = {"Travel": 0.75, "Electronics": 0.50, "Clothing": 0.10,
               "Food":   0.00, "Grocery": -0.20}
    cou_enc = {"NG": 1.35, "TR": 0.95, "FR": 0.10,
               "UK": 0.00, "US": -0.10, "DE": -0.20}

    t_r = np.array([txn_enc[v] for v in transaction_type])
    c_r = np.array([cat_enc[v] for v in merchant_category])
    o_r = np.array([cou_enc[v] for v in country])
    h_r = np.where(hour <= 5, 0.70, np.where(hour >= 22, 0.35, 0.0))
    log_amt = np.log1p(amount)
    a_r = (log_amt - log_amt.mean()) / log_amt.std()

    # Intercept calibrated for expected base fraud rate ≈ 5 %
    logit_context = (
        -3.3
        + 0.8 * t_r
        + 0.7 * c_r
        + 1.0 * o_r
        + 0.8 * h_r
        + 0.5 * a_r
        + rng.standard_normal(n) * 0.8   # contextual noise
    )
    base_scores = _sigmoid(logit_context)

    # ── Stage 2: sample exactly n_fraud fraud cases ──────────────────────────
    fraud_idx = rng.choice(n, size=n_fraud, replace=False,
                           p=base_scores / base_scores.sum())
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_idx] = 1
    fraud_mask = is_fraud.astype(bool)
    legit_mask = ~fraud_mask
    n_legit = legit_mask.sum()

    # ── Stage 3: risk scores conditional on fraud label ──────────────────────
    # Distributions chosen to give AUROC ≈ 0.82 for each score independently
    # and heavy overlap in the 0.10–0.60 range.
    #
    # Legit  – Beta(1.0, 4.0): mean=0.20, sd=0.163  range mostly [0.01, 0.60]
    # Fraud  – Beta(3.0, 4.0): mean=0.43, sd=0.175  range mostly [0.05, 0.80]
    # Overlap: substantial from 0.10 to 0.60

    device_risk_score = np.empty(n)
    device_risk_score[legit_mask] = rng.beta(1.0, 4.0, n_legit)
    device_risk_score[fraud_mask] = rng.beta(3.0, 4.0, n_fraud)

    # ip_risk_score: slightly weaker signal than device
    # Legit  – Beta(1.0, 4.5): mean=0.18, sd=0.154
    # Fraud  – Beta(2.5, 4.0): mean=0.38, sd=0.182
    ip_risk_score = np.empty(n)
    ip_risk_score[legit_mask] = rng.beta(1.0, 4.5, n_legit)
    ip_risk_score[fraud_mask] = rng.beta(2.5, 4.0, n_fraud)

    return pd.DataFrame({
        "transaction_id":    transaction_id,
        "user_id":           user_id,
        "amount":            amount,
        "transaction_type":  transaction_type,
        "merchant_category": merchant_category,
        "country":           country,
        "hour":              hour,
        "device_risk_score": device_risk_score.round(6),
        "ip_risk_score":     ip_risk_score.round(6),
        "is_fraud":          is_fraud,
    })


def main() -> None:
    LOGGER.info("Generating %d transactions (%d fraud, %.0f%%) …",
                N, N_FRAUD, 100 * N_FRAUD / N)
    df = generate()

    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]

    LOGGER.info("Fraud cases: %d (%.1f%%)", len(fraud), 100 * len(fraud) / len(df))
    LOGGER.info("device_risk_score | fraud mean=%.3f sd=%.3f | legit mean=%.3f sd=%.3f",
                fraud["device_risk_score"].mean(), fraud["device_risk_score"].std(),
                legit["device_risk_score"].mean(), legit["device_risk_score"].std())
    LOGGER.info("ip_risk_score     | fraud mean=%.3f sd=%.3f | legit mean=%.3f sd=%.3f",
                fraud["ip_risk_score"].mean(), fraud["ip_risk_score"].std(),
                legit["ip_risk_score"].mean(), legit["ip_risk_score"].std())

    # Verify overlap (legit max > fraud min means no hard gap)
    for col in ["device_risk_score", "ip_risk_score"]:
        legit_max = legit[col].max()
        fraud_min = fraud[col].min()
        legit_min = legit[col].min()
        fraud_max = fraud[col].max()
        LOGGER.info("%s ranges | legit [%.4f, %.4f] | fraud [%.4f, %.4f]",
                    col, legit_min, legit_max, fraud_min, fraud_max)
        if fraud_min > legit_max:
            LOGGER.error("Hard gap detected in %s — review generation logic!", col)
        else:
            LOGGER.info("%s overlap confirmed (legit max %.4f > fraud min %.4f)", col, legit_max, fraud_min)

    LOGGER.info("Saving → %s", RAW_CSV)
    df.to_csv(RAW_CSV, index=False)
    LOGGER.info("Done. Re-run preprocessing then training to use the new dataset.")


if __name__ == "__main__":
    main()
