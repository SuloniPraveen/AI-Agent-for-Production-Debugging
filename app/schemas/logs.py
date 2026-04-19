"""Request/response models for log upload API."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LogUploadResponse(BaseModel):
    """Immediate response after accepting a log upload."""

    batch_id: UUID
    status: str = Field(description="pending until background ingest completes")
    message: str = Field(default="Ingestion started")


class LogBatchStatusResponse(BaseModel):
    """Poll ingestion progress."""

    batch_id: UUID
    status: str
    filename: str
    line_count: int
    chunk_count: int
    error_message: Optional[str] = None


class LogLinesUploadBody(BaseModel):
    """JSON body alternative to multipart file."""

    lines: list[str] = Field(..., description="Raw log lines (any length; large payloads use multipart file)")
    filename: str = Field(default="upload.jsonl", max_length=512)
