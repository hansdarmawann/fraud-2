"""FastAPI server for real-time fraud prediction.

Run with:
    uvicorn code.deployment.api:app --reload --port 8000

Or from code/deployment/:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import MODELS, REPORTS

try:
    from deployment.predict import _load_artifacts, _validate_input, _engineer
except ImportError:
    from predict import _load_artifacts, _validate_input, _engineer

LOGGER = get_logger("api")


class TransactionRequest(BaseModel):
    """Request schema for fraud prediction."""

    amount: float
    hour: int
    device_risk_score: float
    ip_risk_score: float
    transaction_type: str
    merchant_category: str
    country: str


class PredictResponse(BaseModel):
    """Response schema for fraud prediction."""

    fraud_probability: float
    predicted_fraud: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model: str
    trained_at: str


app = FastAPI(title="Fraud Detection API", version="1.0")

_schema: dict | None = None
_encoder = None
_model = None


@app.on_event("startup")
def _startup() -> None:
    global _schema, _encoder, _model
    try:
        _schema, _encoder, _model = _load_artifacts()
        LOGGER.info("Artifacts loaded at startup")
    except Exception as e:
        LOGGER.error("Failed to load artifacts: %s", e)
        raise


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: TransactionRequest) -> PredictResponse:
    """Predict fraud probability for a single transaction."""
    df = pd.DataFrame([req.model_dump()])
    try:
        _validate_input(df, _schema)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    X = _engineer(df, _schema, _encoder)
    proba = float(_model.predict_proba(X)[0, 1])
    predicted = int(proba >= 0.5)
    return PredictResponse(fraud_probability=proba, predicted_fraud=predicted)


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> HealthResponse:
    """Health check endpoint that returns model metadata."""
    metrics_path = REPORTS / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            m = json.load(fp)
        best_estimator = m.get("best_estimator", "unknown")
        trained_at = m.get("trained_at", "unknown")
    else:
        best_estimator = "unknown"
        trained_at = "unknown"
    return HealthResponse(status="ok", model=best_estimator, trained_at=trained_at)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
