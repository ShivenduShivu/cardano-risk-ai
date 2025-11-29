# backend/scripts/batch_score_risk_features.py
"""
Batch score risk_features table using the loaded model.
Writes backend/top_anomalies.csv with top N anomalies.
"""
import os, json
from app.model_loader import get_predictor
from app.supabase_client import get_client
import numpy as np
import pandas as pd

OUTFILE = "top_anomalies.csv"
MAX_ROWS = 50000  # safe cap

def fetch_risk_rows(limit=MAX_ROWS):
    sb = get_client()
    try:
        resp = sb.table("risk_features").select("address, raw").limit(limit).execute()
    except Exception as e:
        print("Supabase fetch failed:", e)
        raise
    rows = getattr(resp, "data", None) or []
    records = []
    for r in rows:
        raw = r.get("raw")
        if raw:
            try:
                rec = json.loads(raw)
            except Exception:
                rec = r
        else:
            rec = r
        rec['address'] = r.get('address') or rec.get('address')
        records.append(rec)
    return pd.DataFrame(records)

def build_feature_matrix(df):
    # expected features order — must match model training
    cols = ['num_txs','num_utxos','num_counterparties','counterparty_ratio',
            'avg_out_amount','small_utxo_count','txs_per_day']
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df['log_avg_out'] = np.log1p(df['avg_out_amount'].fillna(0))
    mat = df[cols + ['log_avg_out']].fillna(0).values
    return mat, df

def score_dataframe(predictor, X):
    # predictor may be pipeline or wrapper
    try:
        # prefer pipeline decision_function
        scores = -predictor.decision_function(X)
    except Exception:
        # try to locate components for scaler/isolationforest
        try:
            scaler = predictor.named_steps.get('standardscaler') if hasattr(predictor, 'named_steps') else None
            if scaler is not None:
                Xs = scaler.transform(X)
            else:
                Xs = X
            est = predictor.named_steps.get('isolationforest') if hasattr(predictor, 'named_steps') else None
            if est is not None:
                scores = -est.decision_function(Xs)
            else:
                # last resort: try estimator on entire pipeline object
                est2 = predictor if not hasattr(predictor, 'named_steps') else list(predictor.named_steps.values())[-1]
                scores = -est2.decision_function(Xs)
        except Exception as e:
            raise RuntimeError("Batch scoring failed: " + str(e))
    return np.ravel(scores)

def main():
    print("Fetching risk_features rows from Supabase...")
    df = fetch_risk_rows()
    if df.empty:
        print("No rows found in risk_features. Exiting.")
        return
    print(f"Loaded {len(df)} rows. Building feature matrix...")
    X, df = build_feature_matrix(df)
    print("Loading predictor from model_loader...")
    predictor = get_predictor()
    print("Scoring...")
    scores = score_dataframe(predictor, X)
    df['anomaly_score'] = scores
    out = df.sort_values('anomaly_score', ascending=False).head(200)
    out[['address','anomaly_score']].to_csv(OUTFILE, index=False)
    print(f"Wrote {OUTFILE} with {len(out)} rows.")

if __name__ == "__main__":
    main()
