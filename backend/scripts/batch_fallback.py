import os, sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from app.supabase_client import get_client
from app.predict import _fallback_score_using_population_stats
print('Fetching risk_features (select=*)...')
sb = get_client()
resp = sb.table('risk_features').select('*').limit(50000).execute()
rows = getattr(resp, 'data', None) or []
if not rows:
    print('NO DATA in risk_features.'); raise SystemExit()
records = []
for r in rows:
    rec = dict(r) if isinstance(r, dict) else {}
    raw = r.get('raw') if isinstance(r, dict) else None
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict): rec.update(parsed)
        except Exception: pass
    records.append(rec)
df = pd.DataFrame(records)
def getval(row, keys, default=0):
    for k in keys:
        if k in row and row[k] is not None:
            try: return float(row[k])
            except Exception: pass
    return float(default)
features_list = []
scores = []
for _, r in df.iterrows():
    row = r.to_dict()
    feat = {}
    feat['num_txs'] = getval(row, ['num_txs','tx_count','txs'], 0)
    feat['num_utxos'] = getval(row, ['num_utxos','utxo_count'], 0)
    feat['num_counterparties'] = getval(row, ['num_counterparties','counterparty_count','n_counterparties'], 0)
    feat['counterparty_ratio'] = getval(row, ['counterparty_ratio','cp_ratio'], 0)
    feat['avg_out_amount'] = getval(row, ['avg_out_amount','avg_out','avg_out_amt'], 0)
    feat['small_utxo_count'] = getval(row, ['small_utxo_count','small_utxos'], 0)
    feat['txs_per_day'] = getval(row, ['txs_per_day','txs_day'], 0)
    features_list.append(feat)
    try:
        s = _fallback_score_using_population_stats(feat)
    except Exception:
        # fallback simple heuristic
        s = math.log1p(max(0.0, (feat['avg_out_amount']>1e5)*1.5 + (feat['num_counterparties']>=10)*2.0))
    scores.append(float(s))
df_out = df.copy().reset_index(drop=True)
df_out['anomaly_score'] = scores
out = df_out.sort_values('anomaly_score', ascending=False).head(200)
out[['address','anomaly_score']].to_csv('top_anomalies.csv', index=False)
print('Wrote top_anomalies.csv with', len(out))
