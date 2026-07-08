import os

INDEX_NAME = os.environ.get("INDEX_NAME", "rag_knowledge_base")

LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.015"))
MAX_CONTEXT_DOCS = int(os.environ.get("MAX_CONTEXT_DOCS", "20"))

# Hugging Face Serverless Inference
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_EMBEDDING_API_URL = os.environ.get(
    "HF_EMBEDDING_API_URL",
    "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
)

# LLM Config (Gemini 2.0 Flash via Google AI Studio)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" or "huggingface"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_URL = os.environ.get(
    "LLM_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
)
HF_LLM_MODEL = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Qdrant Config
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Ingestion tuning
INGEST_BATCH_SIZE = int(os.environ.get("INGEST_BATCH_SIZE", "5"))
MAX_EMBED_RETRIES = int(os.environ.get("MAX_EMBED_RETRIES", "3"))