"""This file contains the chat schema for the application."""

import re
from typing import (
    List,
    Literal,
    Optional,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class Message(BaseModel):
    """Message model for chat endpoint.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    # Assistant replies can exceed a few thousand chars (structured sections + citations).
    content: str = Field(..., description="The content of the message", min_length=1, max_length=65535)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class LogCitation(BaseModel):
    """One retrieved log chunk reference returned with chat responses."""

    model_config = {"extra": "ignore"}

    chunk_id: int = Field(..., description="log_chunks.id")
    batch_id: str = Field(..., description="log_batches.id")
    timestamp: Optional[str] = Field(default=None, description="From chunk metadata when present")
    snippet: str = Field(..., description="Short excerpt")
    service: Optional[str] = None
    level: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        focus_log_batch_id: Optional log upload batch to scope vector search to (must belong to the user).
            When omitted, search uses the user's most recently completed upload.
    """

    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    focus_log_batch_id: Optional[UUID] = Field(
        default=None,
        description="Scope log RAG to this upload batch UUID (defaults to latest completed batch for the user)",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        citations: Log chunks cited via search_logs during this turn (if any).
    """

    messages: List[Message] = Field(..., description="List of messages in the conversation")
    citations: List[LogCitation] = Field(
        default_factory=list,
        description="Retrieved log chunk citations from the last agent turn",
    )


class StreamResponse(BaseModel):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")
