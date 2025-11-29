# backend/app/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import requests
from urllib.parse import quote_plus
from app import model_loader
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional, Dict, Any
import math
import statistics
import json

router = APIRouter()

class PredictRequest(BaseModel):
    address: Optional[str] = None
    num_txs: Optional[float] = None
    num_utxos: Optional[float] = None
    num_counterparties: Optional[float] = None
    counterparty_ratio: Optional[float] = None
    avg_out_amount: Optional[float] = None        # optional, used to compute log_avg_out
    flag_many_small_utxos: int = 0
    flag_high_tx_activity: int = 0

# Use a single requests.Session for REST calls
_session = requests.Session()

def fetch_risk_features(address: str):
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SERVICE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not SUPABASE_URL or not SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    headers = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}
    encoded = quote_plus(address)
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/risk_features?select=*&address=eq.{encoded}"
    r = _session.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    return rows[0]


def _vectorize_from_row_or_request(row, req: PredictRequest):
    """
    Produces feature vector in the same order used at training:
    [num_txs, num_utxos, num_counterparties, counterparty_ratio,
     avg_out_amount, small_utxo_count, txs_per_day, log_avg_out]
    Use fallbacks when fields are missing.
    """
    import numpy as _np

    if row is not None:
        num_txs = float(row.get("num_txs", 0) or 0)
        num_utxos = float(row.get("num_utxos", 0) or 0)
        num_counterparties = float(row.get("num_counterparties", 0) or 0)
        counterparty_ratio = float(row.get("counterparty_ratio", 0) or 0)
        avg_out_amount = float(row.get("avg_out_amount", 0) or 0)
        small_utxo_count = int(row.get("small_utxo_count", 0) or 0)
        txs_per_day = float(row.get("txs_per_day", 0) or 0)
        counterparties = row.get("counterparties") or row.get("top_counterparties") or None
    else:
        if req.num_txs is None or req.num_utxos is None or req.num_counterparties is None or req.counterparty_ratio is None:
            raise ValueError("Missing required numeric fields when address not provided")
        num_txs = float(req.num_txs or 0)
        num_utxos = float(req.num_utxos or 0)
        num_counterparties = float(req.num_counterparties or 0)
        counterparty_ratio = float(req.counterparty_ratio or 0)
        avg_out_amount = float(req.avg_out_amount or 0.0)
        small_utxo_count = int(req.flag_many_small_utxos or 0)
        txs_per_day = float(req.flag_high_tx_activity or 0)
        counterparties = None

    log_avg_out = float(_np.log1p(avg_out_amount))

    vec = [
        num_txs,
        num_utxos,
        num_counterparties,
        counterparty_ratio,
        avg_out_amount,
        small_utxo_count,
        txs_per_day,
        log_avg_out
    ]
    features = {
        "num_txs": num_txs,
        "num_utxos": num_utxos,
        "num_counterparties": num_counterparties,
        "counterparty_ratio": counterparty_ratio,
        "avg_out_amount": avg_out_amount,
        "small_utxo_count": small_utxo_count,
        "txs_per_day": txs_per_day,
        "log_avg_out": log_avg_out,
        "counterparties": counterparties
    }
    return _np.array(vec).reshape(1, -1), features


def _score_from_predictor(predictor, x: np.ndarray):
    """
    Return (anomaly_score, raw_score). Higher anomaly_score => more anomalous.
    Handles pipeline, estimator, or {'scaler','estimator'} wrapper.
    """
    import numpy as _np
    try:
        if hasattr(predictor, "decision_function"):
            raw = predictor.decision_function(x)
            raw_score = float(_np.atleast_1d(raw)[0])
            anomaly_score = -raw_score
            return anomaly_score, raw_score
        if hasattr(predictor, "score_samples"):
            raw = predictor.score_samples(x)
            raw_score = float(_np.atleast_1d(raw)[0])
            anomaly_score = -raw_score
            return anomaly_score, raw_score
        if isinstance(predictor, dict) and "estimator" in predictor:
            est = predictor["estimator"]
            scaler = predictor.get("scaler")
            if scaler is not None:
                x_proc = scaler.transform(x)
            else:
                x_proc = x
            if hasattr(est, "decision_function"):
                raw = est.decision_function(x_proc)
                raw_score = float(_np.atleast_1d(raw)[0])
                anomaly_score = -raw_score
                return anomaly_score, raw_score
            if hasattr(est, "score_samples"):
                raw = est.score_samples(x_proc)
                raw_score = float(_np.atleast_1d(raw)[0])
                anomaly_score = -raw_score
                return anomaly_score, raw_score
        if hasattr(predictor, "predict_proba"):
            probs = predictor.predict_proba(x)
            maxp = float(_np.max(probs, axis=1)[0])
            raw_score = maxp
            anomaly_score = 1.0 - maxp
            return anomaly_score, raw_score
    except Exception as e:
        raise RuntimeError(f"Scoring failed: {e}")

    raise RuntimeError("Predictor does not support decision_function/score_samples/predict_proba")


# Helper to load predictor with a timeout to avoid blocking the request
def _get_predictor_with_timeout(timeout_seconds: int = 10):
    """
    Runs get_predictor() in a thread and returns the predictor, or raises RuntimeError on timeout.
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(model_loader.get_predictor)
        try:
            predictor = fut.result(timeout=timeout_seconds)
            return predictor
        except FutureTimeout:
            fut.cancel()
            raise RuntimeError(f"Model load timed out after {timeout_seconds}s")
        except Exception as e:
            raise RuntimeError(f"Model load failed: {e}")


def _get_model_filename_best_effort():
    # best-effort: read the cached filename in model_loader if present
    try:
        fname = getattr(model_loader, "_cached_remote_filename", None)
        if fname:
            return str(fname)
    except Exception:
        pass
    try:
        res = model_loader.load_latest_model()
        if isinstance(res, dict):
            return res.get("filename") or res.get("model_local") or None
    except Exception:
        pass
    return None


def _apply_conservative_heuristics(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Conservative suppression rules to reduce false positives.
    """
    veto = False
    reasons = []
    try:
        num_txs = float(features.get("num_txs", 0) or 0)
        num_utxos = float(features.get("num_utxos", 0) or 0)
        avg_out_amount = float(features.get("avg_out_amount", 0) or 0)
        SMALL_AMT = 1000.0
        if num_txs < 2 and avg_out_amount < SMALL_AMT:
            veto = True
            reasons.append("very_low_activity_and_small_amounts")
        if num_utxos <= 1 and avg_out_amount < SMALL_AMT:
            veto = True
            reasons.append("single_utxo_small_amount")
    except Exception:
        pass
    return {"veto": veto, "reasons": reasons}


def _fetch_population_stats(sample_limit: int = 2000):
    """
    Fetch population sample of features from risk_features table and compute medians and mad.
    Returns dict: {feature: (median, mad_or_iqr)}
    """
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SERVICE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
    if not SUPABASE_URL or not SERVICE_KEY:
        return None
    headers = {"apikey": SERVICE_KEY, "Authorization": f"Bearer {SERVICE_KEY}"}
    cols = "num_txs,num_utxos,num_counterparties,counterparty_ratio,avg_out_amount,small_utxo_count,txs_per_day"
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/risk_features?select={cols}&order=uploaded_at.desc&limit={sample_limit}"
    try:
        r = _session.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        # compute medians and mad (or iqr fallback)
        stats = {}
        for c in cols.split(","):
            vals = []
            for row in rows:
                v = row.get(c)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if not vals:
                stats[c] = (0.0, 1.0)
                continue
            med = float(statistics.median(vals))
            # compute MAD
            diffs = [abs(x - med) for x in vals]
            mad = float(statistics.median(diffs)) if diffs else 0.0
            if mad <= 0:
                # fallback to IQR-like
                try:
                    q1 = np.percentile(vals, 25)
                    q3 = np.percentile(vals, 75)
                    iqr = float(max(q3 - q1, 1.0))
                    scale = iqr / 1.349 if iqr > 0 else 1.0
                except Exception:
                    scale = 1.0
            else:
                scale = mad
            # avoid zero scale
            if scale <= 0:
                scale = 1.0
            stats[c] = (med, scale)
        return stats
    except Exception:
        return None


def _fallback_score_using_population_stats(features: Dict[str, Any]) -> float:
    """
    Conservative fallback scoring:
      - Fetch recent population stats from risk_features.
      - Compute per-feature z-like = max(0, (value - median)/scale)
      - Weighted sum of the positives normalized to produce a score.
    Returns a raw_score (higher -> more anomalous).
    """
    try:
        stats = _fetch_population_stats(sample_limit=2000)
    except Exception:
        stats = None

    # feature weights (conservative)
    weights = {
        "num_txs": 0.15,
        "num_utxos": 0.10,
        "num_counterparties": 0.20,
        "counterparty_ratio": 0.15,
        "avg_out_amount": 0.25,
        "small_utxo_count": 0.05,
        "txs_per_day": 0.10
    }

    # assemble values in same keys as stats
    keys = ["num_txs","num_utxos","num_counterparties","counterparty_ratio","avg_out_amount","small_utxo_count","txs_per_day"]
    z_sum = 0.0
    weight_sum = 0.0
    for k in keys:
        v = float(features.get(k, 0) or 0)
        weight = float(weights.get(k, 0) or 0)
        weight_sum += weight
        if stats and k in stats:
            med, scale = stats[k]
            try:
                z = (v - med) / float(scale)
            except Exception:
                z = 0.0
            # We only care about positive deviations (conservative)
            if z > 0:
                z_sum += weight * z
        else:
            # if no stats available, apply simple thresholds to detect extremes
            # conservative defaults:
            if k == "avg_out_amount":
                if v > 1e6:
                    z_sum += weight * 5.0
                elif v > 1e5:
                    z_sum += weight * 1.5
            elif k == "num_counterparties":
                if v >= 10:
                    z_sum += weight * 2.0
            elif k == "num_txs":
                if v >= 100:
                    z_sum += weight * 1.5

    # normalize: divide by weight_sum and scale down for conservative score
    if weight_sum <= 0:
        weight_sum = 1.0
    raw = z_sum / weight_sum
    # apply a gentle log scaling so extreme z produce larger scores but keep small ones low
    raw_score = float(math.log1p(max(0.0, raw)))
    # map to anomaly_score (higher => more anomalous). Keep sign consistent with model approach.
    # We'll return anomaly_score := raw_score (higher => more anomalous)
    return raw_score


@router.post("/predict")
async def predict(req: PredictRequest):
    # 1) Attempt to obtain predictor (with timeout)
    try:
        predictor = _get_predictor_with_timeout(timeout_seconds=10)
    except RuntimeError as e:
        # don't fail immediately: we will still attempt fallback scoring
        predictor = None
        loader_error = str(e)
    else:
        loader_error = None

    # 2) Fetch row if address provided
    row = None
    if req.address:
        try:
            row = fetch_risk_features(req.address)
        except Exception as e:
            # If we cannot fetch row, still allow request with provided numeric fields
            row = None

        if req.address and not row and (req.num_txs is None):
            # only error if user provided address but it's missing in DB AND user did not provide fallback numeric fields
            raise HTTPException(status_code=404, detail="Address not found in risk_features; call backfill first or provide numeric fields.")

    # 3) Validate when no address
    if not req.address:
        missing = [f for f in ["num_txs","num_utxos","num_counterparties","counterparty_ratio"] if getattr(req, f) is None]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")

    # 4) Build feature vector + features dict
    try:
        x, features = _vectorize_from_row_or_request(row, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature vectorization failed: {e}")

    # 5) Try scoring with predictor if available; otherwise fallback
    anomaly_score = None
    raw_score = None
    model_used = None
    scoring_error = None

    if predictor is not None:
        try:
            anomaly_score, raw_score = _score_from_predictor(predictor, x)
            model_used = _get_model_filename_best_effort()
        except Exception as e:
            scoring_error = str(e)

    # If predictor scoring failed or predictor not present, use fallback heuristic scoring
    if anomaly_score is None:
        try:
            raw_score = _fallback_score_using_population_stats(features)
            anomaly_score = float(raw_score)
            model_used = model_used or "fallback_population_stats"
        except Exception as e:
            # last-resort: simple deterministic rule-based score
            scoring_error = scoring_error or str(e)
            # simple rules:
            s = 0.0
            if features.get("avg_out_amount", 0) > 1e6:
                s += 2.0
            if features.get("num_counterparties", 0) >= 10:
                s += 1.0
            raw_score = float(s)
            anomaly_score = raw_score
            model_used = model_used or "fallback_simple_rules"

    # 6) Apply conservative heuristics to veto obvious false positives
    heuristic_result = _apply_conservative_heuristics(features)
    base_flag = bool(anomaly_score >= 0.5)  # raise bar: require >=0.5 to consider anomalous (tunable)
    is_anomaly = base_flag and (not heuristic_result.get("veto", False))

    # 7) Build explanation & top counterparties summary
    explanation = {"model_flag": base_flag, "heuristic_veto": heuristic_result.get("veto", False),
                   "heuristic_reasons": heuristic_result.get("reasons", []), "feature_alerts": []}
    try:
        if features["avg_out_amount"] > 1e6:
            explanation["feature_alerts"].append("very_high_avg_out_amount")
        if features["num_counterparties"] >= 10:
            explanation["feature_alerts"].append("many_counterparties")
        if features["small_utxo_count"] >= 20:
            explanation["feature_alerts"].append("many_small_utxos")
    except Exception:
        pass

    # Top counterparties parsing (best-effort)
    top_counterparties = None
    try:
        cp = features.get("counterparties")
        if cp:
            if isinstance(cp, str):
                try:
                    cp_parsed = json.loads(cp)
                    top_counterparties = cp_parsed if isinstance(cp_parsed, list) else [cp_parsed]
                except Exception:
                    top_counterparties = [x.strip() for x in cp.split(",") if x.strip()]
            elif isinstance(cp, list):
                top_counterparties = cp[:5]
            else:
                top_counterparties = [str(cp)]
    except Exception:
        top_counterparties = None

    # 8) Return final JSON
    return {
        "anomaly_score": float(anomaly_score),
        "raw_score": float(raw_score),
        "is_anomaly": bool(is_anomaly),
        "model_filename": model_used,
        "features": features,
        "explanation": explanation,
        "top_counterparties": top_counterparties,
        "loader_error": loader_error,
        "scoring_error": scoring_error
    }
