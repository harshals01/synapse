"""
Embedding service — Local in-memory inference via sentence-transformers.
Uses 'sentence-transformers/all-MiniLM-L6-v2' (384-dim, cosine similarity).
No external API calls, no authentication required.
"""
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model singleton into memory."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' into memory...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def get_embedding(text: str) -> list[float]:
    """
    Generate a single 384-dimensional normalized vector embedding for a text string.
    Runs entirely in-process — no network calls.
    """
    model = get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate normalized 384-dimensional vector embeddings for a list of text strings.
    Processed in parallel batches of 32 using PyTorch — no network calls.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
    return vectors.tolist()
