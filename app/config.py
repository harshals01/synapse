import os
from dotenv import load_dotenv

load_dotenv(override=True)

INDEX_NAME = os.environ.get("INDEX_NAME", "rag_knowledge_base")

LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.005"))
MAX_CONTEXT_DOCS = int(os.environ.get("MAX_CONTEXT_DOCS", "20"))
CHAT_HISTORY_WINDOW: int = int(os.environ.get("CHAT_HISTORY_WINDOW", "6"))


GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")


INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "32"))


API_ACCESS_KEY: str = os.environ.get("API_ACCESS_KEY", "")


_raw_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_allowed_origins.split(",") if o.strip()]
    if _raw_allowed_origins
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
