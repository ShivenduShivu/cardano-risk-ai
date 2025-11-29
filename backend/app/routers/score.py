# backend/app/routers/score.py
from fastapi import APIRouter, HTTPException
from ..blockfrost_client import fetch_tx_or_mock
from ..model_loader import ModelLoader
import logging

router = APIRouter()
logger = logging.getLogger("score-router")
model_loader = ModelLoader()

@router.get("/internal/health_model")
def model_health():
    ok = model_loader.model_loaded
    return {"model_loaded": ok}

@router.get("/score/tx/{tx_hash}")
def score_tx_route(tx_hash: str):
    tx = fetch_tx_or_mock(tx_hash)
    if tx is None:
        raise HTTPException(status_code=502, detail="No tx available")
    features = model_loader.extract_features_from_tx(tx)
    score, label = model_loader.score_features(features)
    return {
        "tx_hash": tx_hash,
        "features": features,
        "anomaly_score": score,
        "risk_label": label,
        "metadata": tx.get("metadata", {}),
    }
