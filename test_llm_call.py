import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_LLM_MODEL = os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

url = "https://router.huggingface.co/v1/chat/completions"
headers = {"Content-Type": "application/json"}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

payload = {
    "model": HF_LLM_MODEL,
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.3
}

print(f"Testing URL: {url}")
print(f"Model: {HF_LLM_MODEL}")
print(f"Token length: {len(HF_TOKEN)}")

res = requests.post(url, headers=headers, json=payload, timeout=60)
print(f"Status Code: {res.status_code}")
print(f"Response Body: {res.text}")
