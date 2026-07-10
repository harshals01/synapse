from dotenv import load_dotenv

load_dotenv(override=True)  # Load .env before any config or service module is imported

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.routes.chat import router as chat_router
from app.routes.ingest import router as ingest_router

app = FastAPI(title="Hybrid Conversational RAG API")

# ── Middleware (must be registered BEFORE routers in FastAPI/Starlette) ────────
# Origins are loaded from ALLOWED_ORIGINS env var (comma-separated).
# Defaults to Vite dev server addresses when the variable is not configured.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(ingest_router)