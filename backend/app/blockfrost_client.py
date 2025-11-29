# backend/app/blockfrost_client.py
"""
Blockfrost client helper.

Behavior:
- If BLOCKFROST_KEY env var exists, queries Blockfrost testnet TX endpoints.
- If no key or request fails, returns a deterministic mocked tx JSON (useful for offline/demo).
Beginner explanation (first-time terms):
- Blockfrost: a hosted API service that lets you read Cardano blockchain data via HTTP. We use the testnet (not real ADA).
- Transaction hash (tx_hash): unique ID for a transaction on the blockchain.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("blockfrost-client")

BLOCKFROST_KEY = os.getenv("BLOCKFROST_KEY", "").strip()
BLOCKFROST_BASE = "https://cardano-testnet.blockfrost.io/api/v0"

MOCK_TX = {
    "hash": "mock_tx_123",
    "inputs": [{"address": "addr_test1...", "amount": [{"unit": "lovelace", "quantity": "1000000"}], "tx_hash": "in_tx"}],
    "outputs": [{"address": "addr_test1...", "amount": [{"unit": "lovelace", "quantity": "900000"}], "tx_hash": "out_tx"}],
    "metadata": {
        "1": {"message": "hello", "value": 42},
        "721": {"policy": "0xabcd", "asset": "MYTOKEN", "meta": {"name": "Test NFT"}}
    },
    "fee": "170000",
    "block_height": 123456,
    "metadata_bytes": 128
}


def _headers():
    return {"project_id": BLOCKFROST_KEY} if BLOCKFROST_KEY else {}


def fetch_tx_or_mock(tx_hash: str) -> Optional[Dict[str, Any]]:
    """
    Try Blockfrost: GET /txs/{tx_hash}
    Also fetch metadata via /txs/{tx_hash}/metadata, and utxo via /txs/{tx_hash}/utxos
    If any fetch fails or no key, return a mocked tx structure.
    """
    if not BLOCKFROST_KEY:
        logger.warning("BLOCKFROST_KEY not set — using mock tx")
        mock = MOCK_TX.copy()
        mock["hash"] = tx_hash
        return mock

    try:
        # tx details
        tx_resp = requests.get(f"{BLOCKFROST_BASE}/txs/{tx_hash}", headers=_headers(), timeout=10)
        if tx_resp.status_code != 200:
            logger.warning("Blockfrost tx fetch failed: %s", tx_resp.text)
            return _mock_for(tx_hash)

        tx_json = tx_resp.json()

        # utxo (inputs/outputs)
        utxo_resp = requests.get(f"{BLOCKFROST_BASE}/txs/{tx_hash}/utxos", headers=_headers(), timeout=10)
        utxo = utxo_resp.json() if utxo_resp.status_code == 200 else {}

        # metadata
        meta_resp = requests.get(f"{BLOCKFROST_BASE}/txs/{tx_hash}/metadata", headers=_headers(), timeout=10)
        metadata = {}
        if meta_resp.status_code == 200:
            # metadata is a list of {label, json_metadata}
            try:
                for item in meta_resp.json():
                    lbl = str(item.get("label"))
                    metadata[lbl] = item.get("json_metadata")
            except Exception:
                metadata = {"raw": meta_resp.text}

        # assemble a compact structure
        assembled = {
            "hash": tx_json.get("hash"),
            "fee": tx_json.get("fee"),
            "block_height": tx_json.get("block_height"),
            "inputs": utxo.get("inputs", []),
            "outputs": utxo.get("outputs", []),
            "metadata": metadata,
            "metadata_bytes": tx_json.get("metadata_bytes", 0)
        }
        return assembled
    except Exception as e:
        logger.exception("Exception fetching from Blockfrost: %s", e)
        return _mock_for(tx_hash)


def _mock_for(tx_hash: str):
    mock = MOCK_TX.copy()
    mock["hash"] = tx_hash
    return mock
