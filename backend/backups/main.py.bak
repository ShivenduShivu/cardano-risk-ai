from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cardano Risk AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; lock this down for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Cardano Risk API Online"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend"}
