import requests
from app import config

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
            
            gemini_payload = {
                "contents": contents
            }
            if system_instruction_parts:
                gemini_payload["systemInstruction"] = {
                    "parts": system_instruction_parts
                }
                
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=gemini_payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API Error: {response.text} (Status Code: {response.status_code})")
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
            # Call Hugging Face OpenAI-compatible Chat Completion API
            url = "https://router.huggingface.co/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if config.HF_TOKEN:
                headers["Authorization"] = f"Bearer {config.HF_TOKEN}"
                
            hf_payload = {
                "model": config.HF_LLM_MODEL,
                "messages": payload.get("messages", []),
                "temperature": 0.3,
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=hf_payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Hugging Face LLM API Error: {response.text} (Status Code: {response.status_code})")
                return "Error communicating with Hugging Face LLM API"
                
            data = response.json()
            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            return reply or "No response from Hugging Face LLM"
            
        else:
            response = requests.post(
                config.LLM_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": ""
                },
                json=payload,
                timeout=1000,
                verify=False
            )

            data = response.json()
            reply = (
                data.get("data", {})
                .get("result", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            return reply or "No response from LLM"

    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        return "Error communicating with LLM"