from pydantic import BaseModel

class ChatRequest(BaseModel):
    messages: list
    top_k: int = 20
