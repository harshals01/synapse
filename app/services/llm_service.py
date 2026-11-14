"""
LLM service — Groq OpenAI-compatible chat completions API.
Model is configured via the GROQ_MODEL environment variable.
"""
import time

import requests

from app import config

_GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """
    Ensure at most ONE system message exists at the start of the message list.
    Combines multiple system messages into a single system message to ensure
    compatibility with OpenAI-style router standards.
    """
    system_parts = []
    other_messages = []

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "").strip()
            if content:
                system_parts.append(content)
        else:
            other_messages.append(msg)

    if not system_parts:
        return other_messages

    combined_system = "\n\n".join(system_parts)
    return [{"role": "system", "content": combined_system}] + other_messages


def _call_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    logger,
    max_retries: int = 3,
) -> requests.Response:
    """
    POST to ``url`` with exponential backoff on HTTP 429 (rate limit).
    Raises ``RuntimeError`` after all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            logger.warning(f"Groq API timeout (attempt {attempt + 1}/{max_retries}).")
            if attempt == max_retries - 1:
                raise TimeoutError(f"Groq API timed out after {max_retries} attempts.")
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", 2 ** attempt))
            logger.warning(
                f"Groq API rate limited (429). Retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait)
            continue

        return response

    raise RuntimeError(f"Groq API failed after {max_retries} retries.")


def call_llm(payload: dict, logger) -> str:
    """
    Send a chat completion request to the Groq OpenAI-compatible API
    and return the generated reply string.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in the environment.")

    headers: dict = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
    }

    messages = _normalize_messages(payload.get("messages", []))
    groq_payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }

    response = _call_with_retry(_GROQ_LLM_URL, headers, groq_payload, logger)

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API error HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    reply: str | None = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )

    if not reply:
        raise ValueError("Groq returned an empty response.")

    return reply
