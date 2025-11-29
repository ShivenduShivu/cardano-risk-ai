# backend/app/supabase_client.py
"""
Supabase helper (idempotent).
Requires SUPABASE_URL and SUPABASE_KEY in environment (.env).
Provides resilient upsert for addresses and risk_features.
"""

import os
import json
from supabase import create_client, Client
from typing import Dict, Any, Optional, List

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

_SUPABASE_CLIENT: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment.")
    if not _SUPABASE_CLIENT:
        raise RuntimeError("Supabase client not initialized.")
    return _SUPABASE_CLIENT

def insert_address(address: str) -> Dict[str, Any]:
    """
    Idempotent insert for addresses: use upsert on conflict.
    Returns dict with status and data (list of rows).
    """
    sb = get_client()
    try:
        res = sb.table("addresses").upsert({"address": address}, on_conflict="address").execute()
        return {"status": getattr(res, "status_code", None), "data": res.data}
    except Exception as e:
        # fallback attempt: try to select existing row
        try:
            existing = sb.table("addresses").select("*").eq("address", address).limit(1).execute()
            return {"status": getattr(existing, "status_code", None), "data": existing.data}
        except Exception:
            raise RuntimeError(f"Failed to insert or fetch address: {e}")

def insert_risk_features(features: Dict[str, Any]) -> Dict[str, Any]:
    sb = get_client()
    payload = {
        "address": features.get("address"),
        "num_txs": features.get("num_txs"),
        "num_utxos": features.get("num_utxos"),
        "num_counterparties": features.get("num_counterparties"),
        "counterparty_ratio": features.get("counterparty_ratio"),
        "flag_many_small_utxos": features.get("flag_many_small_utxos"),
        "flag_high_tx_activity": features.get("flag_high_tx_activity"),
        "raw": json.dumps(features),
    }
    try:
        # upsert risk_features on address to update if present
        res = sb.table("risk_features").upsert(payload, on_conflict="address").execute()
        return {"status": getattr(res, "status_code", None), "data": res.data}
    except Exception as e:
        raise RuntimeError(f"Failed to insert/upsert risk_features: {e}")

def insert_transactions(tx_hashes: List[str]) -> Dict[str, Any]:
    if not tx_hashes:
        return {"status": None, "data": []}
    sb = get_client()
    try:
        existing = sb.table("transactions").select("tx_hash").in_("tx_hash", tx_hashes).execute()
        existing_hashes = set()
        if existing and existing.data:
            existing_hashes = set([r["tx_hash"] for r in existing.data if "tx_hash" in r])
    except Exception:
        existing_hashes = set()

    new_hashes = [h for h in tx_hashes if h not in existing_hashes]
    if not new_hashes:
        return {"status": None, "data": []}

    payload = [{"tx_hash": tx} for tx in new_hashes]
    try:
        res = sb.table("transactions").insert(payload).execute()
        return {"status": getattr(res, "status_code", None), "data": res.data}
    except Exception as e:
        # try chunked inserts as fallback
        try:
            inserted = []
            for i in range(0, len(payload), 100):
                chunk = payload[i:i+100]
                r = sb.table("transactions").insert(chunk).execute()
                if r and getattr(r, "data", None):
                    inserted.extend(r.data)
            return {"status": 207, "data": inserted}
        except Exception:
            raise RuntimeError(f"Failed to insert transactions: {e}")
