# backend/app/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    """
    Simple health check.
    """
    return {"status": "ok", "service": "cardano-risk-ai", "version": "mvp-1"}
