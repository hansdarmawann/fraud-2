"""Central path registry. Importing this module guarantees folders exist."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
FIG_UNI = FIGURES / "univariate"
FIG_MULTI = FIGURES / "multivariate"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"

RAW_CSV = DATA_RAW / "synthetic_fraud_dataset.csv"
TARGET_COL = "is_fraud"
ID_COLS = ["transaction_id", "user_id"]

for _p in (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_EXTERNAL,
    FIGURES,
    FIG_UNI,
    FIG_MULTI,
    MODELS,
    REPORTS,
    LOGS,
):
    _p.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load pipeline configuration from config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
