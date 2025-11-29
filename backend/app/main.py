# backend/app/main.py
"""
FastAPI backend for Cardano risk MVP (fixed routing & robust frontend path resolution).

This version computes the repo root reliably (two levels up from this file: app -> backend -> repo_root)
and then looks for the repo_root/frontend folder. It logs the exact path and whether it exists.
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .blockfrost_client import fetch_tx_or_mock
from .model_loader import ModelLoader
from .routers.health import router as health_router
from .routers.score import router as score_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cardano-risk-ai")

app = FastAPI(title="Cardano Risk AI (MVP)")

# CORS — permissive for hackathon/demo. Narrow this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader early
model_loader = ModelLoader()

# Include API routers (health, score). These must be registered BEFORE static mount to ensure they get precedence.
app.include_router(health_router)
app.include_router(score_router)

# Robustly resolve project root and frontend directory:
# file structure: repo_root/backend/app/main.py -> go two parents up to repo_root
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]   # app -> backend -> repo_root
FRONTEND_DIR = (REPO_ROOT / "frontend").resolve()

logger.info("Resolved paths: THIS_FILE=%s", THIS_FILE)
logger.info("REPO_ROOT=%s", REPO_ROOT)
logger.info("FRONTEND_DIR=%s", FRONTEND_DIR)
logger.info("FRONTEND_DIR.exists=%s", FRONTEND_DIR.exists())

# Mount static files under /static to avoid shadowing API routes
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info("Mounted frontend static files at /static from %s", FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at %s — not mounting static files", FRONTEND_DIR)

# Serve the index.html at root explicitly
INDEX_FILE = FRONTEND_DIR / "index.html"
if INDEX_FILE.exists():
    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(INDEX_FILE))
    logger.info("Serving index.html at / (from %s)", INDEX_FILE)
else:
    logger.warning("index.html not found at %s", INDEX_FILE)

@app.on_event("startup")
def startup_event():
    logger.info("Starting Cardano Risk AI backend")
    if not model_loader.model_loaded:
        logger.info("Model not loaded — attempting load/quick-train")
        model_loader.load_or_train()
    logger.info("Startup complete.")

# Keep explicit API wrappers (these won't be shadowed because we mounted static under /static)
@app.get("/cardano/tx/{tx_hash}")
def get_tx(tx_hash: str):
    tx = fetch_tx_or_mock(tx_hash)
    return tx or {}

@app.get("/score/tx/{tx_hash}")
def score_tx(tx_hash: str):
    tx = fetch_tx_or_mock(tx_hash)
    features = model_loader.extract_features_from_tx(tx)
    score, label = model_loader.score_features(features)
    return {
        "tx_hash": tx_hash,
        "metadata": tx.get("metadata", {}),
        "features": features,
        "anomaly_score": score,
        "risk_label": label,
    }
