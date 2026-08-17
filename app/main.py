from dotenv import load_dotenv

load_dotenv(override=True)  

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS
from app.routes.chat import router as chat_router
from app.routes.documents import router as documents_router
from app.routes.ingest import router as ingest_router

app = FastAPI(title="Hybrid Conversational RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(documents_router)