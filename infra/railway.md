# Railway deployment guide (backend)

We will deploy the FastAPI backend to Railway (free tier).

1. Create Railway account (https://railway.app) — free tier is enough.
2. Create a new Project → Deploy from GitHub → connect to your `cardano-risk-ai` repo and choose the backend folder.
3. Railway will detect Procfile. If not, set the start command:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add Environment Variables in Railway (Project Settings -> Variables):
   - BLOCKFROST_KEY (optional — your Blockfrost testnet project_id)
   - SUPABASE_URL (optional)
   - SUPABASE_SERVICE_KEY (optional)
   - MODEL_PATH (optional, e.g., public/models/isolation_model.joblib)
5. Deploy. Railway gives you a public URL (e.g., https://your-project.up.railway.app).
6. Update frontend on Vercel (or in local frontend) to call that Railway URL if backend is remote.

Troubleshooting:
- If you prefer local demo, run using uvicorn in Codespaces (see README run commands).
