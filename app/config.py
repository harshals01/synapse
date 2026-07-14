import os

INDEX_NAME = os.environ.get("INDEX_NAME", "rag_knowledge_base")

LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.015"))
MAX_CONTEXT_DOCS = int(os.environ.get("MAX_CONTEXT_DOCS", "20"))
# Number of prior conversation turns sent to the LLM as chat history.
CHAT_HISTORY_WINDOW: int = int(os.environ.get("CHAT_HISTORY_WINDOW", "6"))

# ── Hugging Face Serverless Inference — Embeddings ────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_EMBEDDING_API_URL = os.environ.get(
    "HF_EMBEDDING_API_URL",
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
)

# ── Hugging Face OpenAI-Compatible Router — LLM ───────────────────────────────
HF_LLM_MODEL = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# ── Qdrant Cloud ──────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# ── Ingestion Tuning ──────────────────────────────────────────────────────────
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "5"))
MAX_EMBED_RETRIES = int(os.environ.get("MAX_EMBED_RETRIES", "3"))

# ── Security ──────────────────────────────────────────────────────────────────
API_ACCESS_KEY: str = os.environ.get("API_ACCESS_KEY", "")

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins. Set to your deployed frontend URL
# in production. Leave unset or empty to default to Vite dev server addresses.
_raw_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_allowed_origins.split(",") if o.strip()]
    if _raw_allowed_origins
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)