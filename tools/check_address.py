#!/usr/bin/env python3
"""
tools/check_address.py

Usage:
  # in Codespace terminal (from repo root)
  python3 tools/check_address.py

The script will prompt for:
 - Backend base URL (default http://localhost:8000)
 - Whether to run a backfill (y/N)
 - An address to analyze

It then calls:
 - POST /cardano/backfill/{address}  (optional)
 - POST /ml/predict  (with {"address": "<address>"})
 - GET  /rest/v1/risk_features?address=eq.<address> (Supabase PostgREST) to show feature row

Make sure to set env vars in the terminal before running:
 export SUPABASE_URL="https://<your-supabase>.supabase.co"
 export SUPABASE_KEY="<service-role-key>"   # set in terminal only
"""
import os
import requests
import urllib.parse
import json
import sys
import time

# Helper to read input safely
def ask(prompt, default=None):
    v = input(f"{prompt}{' ['+str(default)+']' if default else ''}: ").strip()
    return v if v else default

def print_div():
    print("-" * 72)

def run_backfill(base_url, address):
    url = f"{base_url.rstrip('/')}/cardano/backfill/{urllib.parse.quote(address, safe='')}"
    print(f"Running backfill -> {url}")
    try:
        r = requests.post(url, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Backfill failed:", str(e))
        # show response text if available
        try:
            print("Response text:", r.text[:800])
        except Exception:
            pass
        return None

def predict_address(base_url, address):
    url = f"{base_url.rstrip('/')}/ml/predict"
    payload = {"address": address}
    print(f"Calling ML predict -> {url} with address")
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Predict call failed:", str(e))
        try:
            print("Response text:", r.text[:1000])
        except Exception:
            pass
        return None

def fetch_risk_features_from_supabase(supabase_url, supabase_key, address):
    # Use PostgREST to fetch risk_features row
    headers = {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"}
    # encode address param properly
    params = {"select": "*", "address": f"eq.{address}"}
    url = f"{supabase_url.rstrip('/')}/rest/v1/risk_features"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        return rows[0]
    except Exception as e:
        print("Failed to fetch risk_features:", str(e))
        try:
            print("Response text:", r.text[:800])
        except Exception:
            pass
        return None

def assess(features, predict_json):
    """
    Simple heuristic assessment combining ML output and feature flags.
    Returns (verdict, reasons_list).
    """
    reasons = []
    verdict = "Likely safe"

    # Interpret ML result
    if not predict_json:
        reasons.append("No ML prediction available.")
    else:
        is_anom = predict_json.get("is_anomaly")
        anom_score = predict_json.get("anomaly_score")  # higher -> more anomalous
        raw_score = predict_json.get("raw_score")
        reasons.append(f"ML: anomaly_score={anom_score}, raw_score={raw_score}, is_anomaly={is_anom}")
        if is_anom:
            verdict = "Compromised / High risk"
            reasons.append("Model flagged this address as an anomaly (high risk).")

    if features:
        # heuristics
        if features.get("flag_many_small_utxos"):
            reasons.append("Heuristic: many small UTXOs flag set => possible dusting or automated funds splitting.")
            if verdict == "Likely safe":
                verdict = "Suspicious"
        if features.get("flag_high_tx_activity"):
            reasons.append("Heuristic: high tx activity flag set => unusually active, could be compromise or exchange.")
            if verdict == "Likely safe":
                verdict = "Suspicious"
        cnt = features.get("num_counterparties") or 0
        num_txs = features.get("num_txs") or 0
        ratio = features.get("counterparty_ratio") or 0.0
        reasons.append(f"Features: num_txs={num_txs}, num_utxos={features.get('num_utxos')}, num_counterparties={cnt}, counterparty_ratio={ratio}")
        # suspicious ratio: many counterparties relative to txs
        if ratio > 0.8 and num_txs > 5:
            reasons.append("High counterparty ratio (>0.8) — interacting with many unique addresses quickly.")
            if verdict == "Likely safe":
                verdict = "Suspicious"

    # final consolidation
    if verdict == "Likely safe" and predict_json and predict_json.get("anomaly_score", 0) > 0.5:
        verdict = "Suspicious"
        reasons.append("Anomaly score moderately high (>0.5) — consider manual review.")

    return verdict, reasons

def main():
    print_div()
    print("Cardano Address Quick Risk Check")
    print_div()

    base_url = ask("Backend base URL (example: http://localhost:8000)", "http://localhost:8000")
    run_backfill_choice = ask("Run backfill for the address first? (y/N)", "N").lower() == "y"

    # Ensure SUPABASE env present for fetching features (only for feature display)
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("NOTE: SUPABASE_URL and SUPABASE_KEY are not set in environment. The script will still call /ml/predict but cannot fetch risk_features locally from Supabase.")
        print("If you want feature display, set SUPABASE_URL and SUPABASE_KEY in the terminal before running this script.")
    print_div()

    address = ask("Paste Cardano address to check (paste full address)", "")
    if not address:
        print("No address provided. Exiting.")
        return

    if run_backfill_choice:
        print_div()
        print("Running backfill (may take some time if many pages)...")
        bf = run_backfill(base_url, address)
        print("Backfill result (truncated):")
        try:
            print(json.dumps(bf, indent=2)[:2000])
        except Exception:
            print(str(bf))
        print_div()
        # small wait to ensure DB row present
        time.sleep(2)

    # ML predict call
    pred = predict_address(base_url, address)
    print_div()

    # fetch risk_features via Supabase PostgREST for display
    features = None
    if SUPABASE_URL and SUPABASE_KEY:
        print("Fetching risk_features row from Supabase (if exists)...")
        features = fetch_risk_features_from_supabase(SUPABASE_URL, SUPABASE_KEY, address)
        if features:
            print("risk_features row:")
            print(json.dumps(features, indent=2))
        else:
            print("No risk_features row found in Supabase for this address.")
    else:
        print("Skipping risk_features fetch (no SUPABASE env).")

    print_div()
    verdict, reasons = assess(features, pred)
    print("FINAL VERDICT:", verdict)
    print_div()
    print("Reasons / evidence:")
    for r in reasons:
        print("-", r)
    print_div()

    # Suggest next actions
    print("Suggested follow-ups:")
    if verdict.startswith("Compromised"):
        print("- Immediately freeze/monitor funds if you control the address.")
        print("- Inspect recent transactions (use /cardano/address/{address} endpoints) to find outgoing txs.")
        print("- Identify counterparties via backfill and check their risk status.")
    elif verdict == "Suspicious":
        print("- Manual review recommended: look at recent transactions, amounts, and counterparties.")
        print("- Consider increasing backfill depth (more pages) and re-running ML after more data.")
    else:
        print("- No immediate sign of compromise by heuristics and current model. Still follow up if user reports suspicious activity.")
    print_div()

    # How to get other addresses
    print("How to obtain other wallet addresses for testing:")
    print("1) Use Blockfrost address tx endpoints to discover counterparties for a known address (backfill produces tx list and counterparties).")
    print("2) Query your Supabase `addresses` table (if you collect addresses) via PostgREST.")
    print("3) Use PREPROD faucets or sample test addresses from Cardano developer docs or Blockfrost docs.")
    print("4) Crawl transactions: for each tx, collect input/output addresses and add them to your test set.")
    print_div()

if __name__ == '__main__':
    main()
