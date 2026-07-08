import logging
import time

import requests

from app import config

logger = logging.getLogger(__name__)

# Maximum characters sent per chunk — all-MiniLM-L6-v2 accepts up to ~512 tokens (~2000 chars)
_MAX_CHARS_PER_CHUNK = 2000


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_headers() -> dict:
    headers: dict = {"Content-Type": "application/json"}
    if config.HF_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
    return headers


def _handle_non_200(response: requests.Response, attempt: int) -> None:
    """
    Handle a non-200 HF API response by sleeping before the next retry.
    Raises ValueError immediately for terminal errors (4xx other than 429/413).
    """
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", 2 ** attempt))
        logger.warning(f"HF API rate limited (429). Retrying in {wait}s (attempt {attempt + 1}).")
        time.sleep(wait)
        return

    if response.status_code == 503:
        estimated = response.json().get("estimated_time", 20)
        wait = min(float(estimated), 30.0)
        logger.warning(f"HF model loading (503). Waiting {wait:.0f}s (attempt {attempt + 1}).")
        time.sleep(wait)
        return

    raise ValueError(
        f"HF Inference API error: HTTP {response.status_code} — {response.text}"
    )


def _parse_single_embedding(data: object) -> list[float]:
    """Extract a single 384-dim vector from a HF API response."""
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            return data[0]
        return data  # type: ignore[return-value]
    raise ValueError(f"Unexpected embedding response format: {type(data)}")


# ── Public API ────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Fetch a single 384-dimensional embedding from the HF Serverless Inference API.

    Retries on HTTP 429 (rate limit) and 503 (model cold-start) with exponential
    backoff for up to MAX_EMBED_RETRIES attempts. Input text is silently truncated
    to 2000 characters if it exceeds the soft token limit.
    """
    if len(text) > _MAX_CHARS_PER_CHUNK:
        logger.debug("Chunk truncated from %d to %d chars.", len(text), _MAX_CHARS_PER_CHUNK)
        text = text[:_MAX_CHARS_PER_CHUNK]

    payload = {"inputs": text, "options": {"wait_for_model": True}}

    for attempt in range(config.MAX_EMBED_RETRIES):
        try:
            response = requests.post(
                config.HF_EMBEDDING_API_URL,
                headers=_build_headers(),
                json=payload,
                timeout=30,
            )
        except requests.exceptions.Timeout:
            logger.warning("HF API timeout (attempt %d).", attempt + 1)
            if attempt == config.MAX_EMBED_RETRIES - 1:
                raise TimeoutError(
                    f"HF Inference API timed out after {config.MAX_EMBED_RETRIES} attempts."
                )
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return _parse_single_embedding(response.json())

        _handle_non_200(response, attempt)

    raise RuntimeError(
        f"HF Embedding API failed after {config.MAX_EMBED_RETRIES} retries."
    )


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks in a single HF API call.

    Each chunk is truncated to 2000 characters before sending. Falls back
    automatically to serial embedding if the server returns HTTP 413
    (payload too large) or if all batch retries are exhausted.
    """
    truncated = [
        t[:_MAX_CHARS_PER_CHUNK] if len(t) > _MAX_CHARS_PER_CHUNK else t
        for t in texts
    ]
    payload = {"inputs": truncated, "options": {"wait_for_model": True}}

    for attempt in range(config.MAX_EMBED_RETRIES):
        try:
            response = requests.post(
                config.HF_EMBEDDING_API_URL,
                headers=_build_headers(),
                json=payload,
                timeout=60,
            )
        except requests.exceptions.Timeout:
            logger.warning("HF batch API timeout (attempt %d).", attempt + 1)
            if attempt == config.MAX_EMBED_RETRIES - 1:
                logger.warning("Falling back to serial embedding after timeout.")
                return [get_embedding(t) for t in texts]
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 413:
            logger.warning("Batch payload too large (413). Falling back to serial embedding.")
            return [get_embedding(t) for t in texts]

        _handle_non_200(response, attempt)

    logger.warning("Batch embedding failed after %d retries. Falling back to serial.", config.MAX_EMBED_RETRIES)
    return [get_embedding(t) for t in texts]