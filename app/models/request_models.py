from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation turn."""
    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Must be one of: user, assistant, system",
    )
    content: str = Field(
        ...,
        description="Message text content",
    )


class ChatRequest(BaseModel):
    """Request body for the POST /chat endpoint."""
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history. Must contain at least one message.",
    )
    top_k: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum number of vector search results to retrieve.",
    )
    document_filter: str = Field(
        "latest",
        description=(
            "Controls which documents are searched. "
            "'latest' (default) searches only the most recently ingested document. "
            "'all' searches across all uploaded documents. "
            "Any other value is treated as an exact source_file name to filter by."
        ),
    )
