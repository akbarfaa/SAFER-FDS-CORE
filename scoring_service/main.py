"""
SAFER Scoring Service — FastAPI Application

Receives raw transactions, scores them via XGBoost and LightGBM ensemble,
computes SHAP values, and outputs explanatory reasoning.
Runs on port 8001.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any

from shared.config import CORS_ORIGINS, SCORING_PORT
from scoring_service.ml_engine import engine
from scoring_service.explainer import SHAPExplainer

app = FastAPI(
    title="SAFER ML Scoring & Explanation Service",
    description="Ensemble XGBoost + LightGBM fraud risk classification and SHAP explainer.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize explainer
explainer = SHAPExplainer(engine.xgb_model)


class TransactionPayload(BaseModel):
    """Payload representing a single raw transaction for scoring."""
    id: Optional[str] = None
    sender_name: Optional[str] = None
    sender_account: Optional[str] = None
    sender_bank: Optional[str] = None
    sender_city: Optional[str] = None
    sender_province: Optional[str] = None
    sender_lat: Optional[float] = 0.0
    sender_lng: Optional[float] = 0.0
    receiver_name: Optional[str] = None
    receiver_account: Optional[str] = None
    receiver_bank: Optional[str] = None
    receiver_city: Optional[str] = None
    receiver_province: Optional[str] = None
    receiver_lat: Optional[float] = 0.0
    receiver_lng: Optional[float] = 0.0
    amount: float
    payment_rail: str
    ewallet_provider: Optional[str] = "None"
    merchant: Optional[str] = None
    merchant_category: Optional[str] = None
    channel: Optional[str] = None
    device_type: Optional[str] = None
    device_brand: Optional[str] = None
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    is_new_device: Optional[int] = 0
    account_age_days: Optional[int] = 365
    is_velocity_anomaly: Optional[int] = 0
    is_geo_mismatch: Optional[int] = 0
    is_off_hours: Optional[int] = 0
    is_high_value_for_rail: Optional[int] = 0
    is_suspicious_ip: Optional[int] = 0
    is_risky_merchant: Optional[int] = 0
    is_new_account: Optional[int] = 0
    has_failed_attempts: Optional[int] = 0
    is_device_mismatch: Optional[int] = 0
    is_sim_swap: Optional[int] = 0
    is_unusual_beneficiary: Optional[int] = 0
    velocity_count: Optional[int] = 1
    geo_distance_km: Optional[float] = 0.0


@app.post("/score")
def score_transaction(payload: TransactionPayload):
    """
    Classify transaction risk using the ML models and generate a SHAP explanation.
    """
    raw_tx = payload.model_dump()

    # Step 1: Score the transaction via ML ensemble
    scoring_result = engine.score_transaction(raw_tx)

    # Step 2: Extract features df and compute SHAP explanation
    features_df = scoring_result.pop("features_df", None)
    
    explanation = explainer.explain(
        tx=raw_tx,
        features_df=features_df,
        risk_score=scoring_result["risk_score"],
        severity=scoring_result["severity"],
        prob=scoring_result["fraud_probability"]
    )

    # Combine scoring and explanation
    response = {
        "risk_score": scoring_result["risk_score"],
        "severity": scoring_result["severity"],
        "fraud_probability": float(scoring_result["fraud_probability"]),
        "xgb_probability": float(scoring_result["xgb_probability"]),
        "lgb_probability": float(scoring_result["lgb_probability"]),
        "ai_reasoning": explanation["ai_reasoning"],
        "shap_values": explanation["shap_values"],
        "primary_risk_factors": explanation["primary_risk_factors"],
        "suggested_action": explanation["suggested_action"],
    }

    return response


@app.get("/health")
def health():
    return {
        "service": "scoring",
        "status": "ok",
        "models_loaded": engine.is_loaded,
        "port": SCORING_PORT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scoring_service.main:app", host="0.0.0.0", port=SCORING_PORT, reload=True)
