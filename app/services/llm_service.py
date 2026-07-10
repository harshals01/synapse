"""
LLM service — Hugging Face OpenAI-compatible chat completions router.

The application uses the Hugging Face serverless inference router exclusively.
Model is configured via the HF_LLM_MODEL environment variable.
"""
import time

import requests

from app import config

_HF_LLM_URL = "https://router.huggingface.co/v1/chat/completions"


def _call_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    logger,
    max_retries: int = 3,
) -> requests.Response:
    """
    POST to ``url`` with exponential backoff on HTTP 429 (rate limit) and
    503 (model loading). Raises ``RuntimeError`` after all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            logger.warning(f"LLM API timeout (attempt {attempt + 1}/{max_retries}).")
            if attempt == max_retries - 1:
                raise TimeoutError(f"LLM API timed out after {max_retries} attempts.")
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            logger.warning(
                f"LLM API rate limited (429). Retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait)
            continue

        if response.status_code == 503:
            body = response.json() if response.content else {}
            wait = min(float(body.get("estimated_time", 20)), 30.0)
            logger.warning(
                f"LLM model loading (503). Waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait)
            continue

        # Non-retryable error — return immediately so the caller can log it
        return response

    raise RuntimeError(f"LLM API failed after {max_retries} retries.")


def call_llm(payload: dict, logger) -> str:
    """
    Send a chat completion request to the Hugging Face OpenAI-compatible
    router and return the generated reply string.

    Retries automatically on HTTP 429 (rate limit) and 503 (model cold-start).
    Returns a human-readable error string on persistent failure so the caller
    can surface it to the user gracefully.
    """
    try:
        headers: dict = {"Content-Type": "application/json"}
        if config.HF_TOKEN:
            headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

        hf_payload = {
            "model": config.HF_LLM_MODEL,
            "messages": payload.get("messages", []),
            "temperature": 0.3,
        }

        response = _call_with_retry(_HF_LLM_URL, headers, hf_payload, logger)

        if response.status_code != 200:
            logger.error(
                f"HF LLM API error: HTTP {response.status_code} — {response.text}"
            )
            return "Error communicating with the language model. Please try again."

        data = response.json()
        reply: str | None = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        return reply or "No response was returned by the language model."

    except Exception:
        logger.exception("Unhandled exception in call_llm")
        return "Error communicating with the language model."