# Cardano Risk AI — Hackathon MVP(There are some issues at the moment in the deployment I am trying to sort it out Soon)

Quick start checklist (Codespaces, offline-friendly):

## Required env vars (local / Codespaces / Railway)
- BLOCKFROST_KEY  (optional) — your Blockfrost testnet project id. Signup: https://blockfrost.io
- SUPABASE_URL    (optional) — e.g. https://xyz.supabase.co
- SUPABASE_SERVICE_KEY (optional) — Supabase service role key to upload/download model
- MODEL_PATH (optional) — e.g. public/models/isolation_model.joblib

If none of Supabase/Blockfrost are provided, the backend uses mocks and trains a tiny model automatically.

## Run locally in Codespaces (recommended for demo)
Open the project in GitHub Codespaces (or any environment with Python 3.11+).

1. Install dependencies (devcontainer may have done this):
-cd backend
-pip install -r requirements.txt


2. Start backend:
export BLOCKFROST_KEY= # optional
export SUPABASE_URL= # optional
export SUPABASE_SERVICE_KEY= # optional
uvicorn app.main:app --host 0.0.0.0 --port 8000


3. Open the frontend:
- Open `frontend/index.html` in Codespaces port preview or serve statically:
  ```
  npx serve frontend
  ```
- Or open directly in browser if you deployed to Vercel.

## Endpoints
- GET /health
- GET /cardano/tx/{tx_hash}
- GET /score/tx/{tx_hash}

## Quick offline demo
- Start backend (no env vars). Backend trains fallback model and uses mocked tx data.
- Open frontend and press "Fetch & Score" (uses mock tx).

## Colab: train & upload model
See `colab/train_and_upload.ipynb` for code cells to train IsolationForest and upload to Supabase Storage.

## Security notes
- Keep SUPABASE_SERVICE_KEY secret (store in Railway secret or environment variable).
- Blockfrost project_id is not critical but keep private.
- CORS is permissive here for a demo — restrict in production.

Good luck — we can iterate next to add explainability, model calibration, and more features.
