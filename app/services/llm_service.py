import time
import requests
from app import config


def _call_with_retry(url: str, headers: dict, payload: dict, logger, max_retries: int = 3) -> requests.Response:
    """
    POST to `url` with exponential backoff on HTTP 429 (rate limit) and
    503 (model loading). Raises RuntimeError after all retries are exhausted.
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
            logger.warning(f"LLM API rate limited (429). Retrying in {wait}s (attempt {attempt + 1}).")
            time.sleep(wait)
            continue

        if response.status_code == 503:
            estimated = response.json().get("estimated_time", 20) if response.content else 20
            wait = min(float(estimated), 30.0)
            logger.warning(f"LLM model loading (503). Waiting {wait:.0f}s (attempt {attempt + 1}).")
            time.sleep(wait)
            continue

        # Non-retryable error — return immediately so caller can log and handle
        return response

    raise RuntimeError(f"LLM API failed after {max_retries} retries.")


def call_llm(payload, logger):
    try:
        if config.LLM_PROVIDER == "gemini" and config.GEMINI_API_KEY:
            url = f"{config.LLM_URL}?key={config.GEMINI_API_KEY}"

            contents = []
            system_instruction_parts = []

            for msg in payload.get("messages", []):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction_parts.append({"text": content})
                elif role == "user":
                    contents.append({"role": "user", "parts": [{"text": content}]})
                elif role in ("assistant", "model"):
                    contents.append({"role": "model", "parts": [{"text": content}]})

            gemini_payload = {"contents": contents}
            if system_instruction_parts:
                gemini_payload["systemInstruction"] = {"parts": system_instruction_parts}

            headers = {"Content-Type": "application/json"}
            response = _call_with_retry(url, headers, gemini_payload, logger)

            if response.status_code != 200:
                logger.error(f"Gemini API Error: {response.text} (Status: {response.status_code})")
                return "Error communicating with Gemini API"

            data = response.json()
            reply = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text")
            )
            return reply or "No response from Gemini API"

        elif config.LLM_PROVIDER == "huggingface":
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if config.HF_TOKEN:
                headers["Authorization"] = f"Bearer {config.HF_TOKEN}"

            hf_payload = {
                "model": config.HF_LLM_MODEL,
                "messages": payload.get("messages", []),
                "temperature": 0.3,
            }

            response = _call_with_retry(url, headers, hf_payload, logger)

            if response.status_code != 200:
                logger.error(f"HF LLM API Error: {response.text} (Status: {response.status_code})")
                return "Error communicating with Hugging Face LLM API"

            data = response.json()
            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            return reply or "No response from Hugging Face LLM"

        else:
            logger.error("LLM_PROVIDER is not configured to a supported value ('gemini' or 'huggingface').")
            return "LLM provider not configured."

    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        return "Error communicating with LLM"