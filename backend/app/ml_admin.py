# backend/app/ml_admin.py
from fastapi import APIRouter, HTTPException
import os
import requests
from typing import Any

router = APIRouter()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

def _supabase_get(path: str, params: dict | None = None) -> Any:
    url = f"{SUPABASE_URL.rstrip('/')}{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        raise HTTPException(status_code=r.status_code, detail=f"Supabase GET failed: {r.text[:300]}")
    try:
        return r.json()
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

def _supabase_post(path: str, payload: dict, prefer_return=True) -> Any:
    url = f"{SUPABASE_URL.rstrip('/')}{path}"
    headers = {**HEADERS, "Content-Type": "application/json"}
    if prefer_return:
        headers["Prefer"] = "return=representation"
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        raise HTTPException(status_code=r.status_code, detail=f"Supabase POST failed: {r.text[:300]}")
    try:
        return r.json()
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

@router.get("/models", summary="List models (proxy to Supabase models table)")
def list_models(limit: int = 10):
    params = {"select": "*", "order": "uploaded_at.desc", "limit": limit}
    return _supabase_get("/rest/v1/models", params=params)

@router.post("/models/reload", summary="Force model_loader to reload latest model")
def reload_model():
    """
    Forcibly clear model cache in model_loader (if present) and attempt to download/load latest model.
    Returns best-effort result and does not crash the server if loader lacks a reload function.
    """
    try:
        import importlib
        import app.model_loader as model_loader
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not import model_loader: {str(e)}")

    # Defensive: clear cached filename attribute if present
    if hasattr(model_loader, "_cached_filename"):
        try:
            setattr(model_loader, "_cached_filename", None)
        except Exception:
            pass

    # If module exposes a reload function, call it
    for fn_name in ("load_latest_model", "reload_model", "download_and_cache_latest_model", "load_model"):
        fn = getattr(model_loader, fn_name, None)
        if callable(fn):
            try:
                result = fn()
                return {"status": "reloaded", "detail": str(result)}
            except Exception as e:
                return {"status": "error", "detail": f"called {fn_name} and it raised: {str(e)}"}

    # Fallback: tell user to restart if no reload function exists
    return {"status": "noop", "detail": "No reload function found in model_loader. Please restart the backend (uvicorn) to load the new model."}
