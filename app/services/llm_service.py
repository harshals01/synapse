"""
LLM service — Hugging Face OpenAI-compatible chat completions router.

The application uses the Hugging Face serverless inference router exclusively.
Model is configured via the HF_LLM_MODEL environment variable.
"""
import time

import requests

from app import config

_HF_LLM_URL = "https://router.huggingface.co/v1/chat/completions"

_FALLBACK_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct:fastest",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct:fastest",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct:fastest",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]


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

        return response

    raise RuntimeError(f"LLM API failed after {max_retries} retries.")


def call_llm(payload: dict, logger) -> str:
    """
    Send a chat completion request to the Hugging Face OpenAI-compatible
    router and return the generated reply string.

    Normalizes system messages, enforces token-friendly structure, and retries
    with fallback models if the primary model encounters errors.
    """
    try:
        headers: dict = {"Content-Type": "application/json"}
        if config.HF_TOKEN:
            headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

        messages = _normalize_messages(payload.get("messages", []))

        models_to_try = [config.HF_LLM_MODEL]
        for fallback in _FALLBACK_MODELS:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_error = ""

        for model in models_to_try:
            hf_payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
            }

            try:
                response = _call_with_retry(_HF_LLM_URL, headers, hf_payload, logger)

                if response.status_code == 200:
                    data = response.json()
                    reply: str | None = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    if reply:
                        return reply

                last_error = f"HTTP {response.status_code} — {response.text}"
                logger.warning(
                    f"HF LLM model '{model}' failed: {last_error}"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Exception trying model '{model}': {e}")

        logger.error(f"All LLM models failed. Last error: {last_error}")
        return f"Error communicating with the language model ({last_error}). Please try again."

    except Exception as exc:
        logger.exception("Unhandled exception in call_llm")
        return f"Error communicating with the language model: {exc}"