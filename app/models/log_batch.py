"""Upload batch metadata for log ingestion."""

from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel

from app.models.base import BaseModel


class LogBatchStatus(str, Enum):
    """Background ingestion lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class LogBatch(BaseModel, table=True):
    """One uploaded log file or JSON-lines payload."""

    __tablename__ = "log_batches"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    filename: str = Field(default="", max_length=512)
    status: str = Field(default=LogBatchStatus.PENDING.value, max_length=32, index=True)
    line_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = Field(default=None, max_length=4000)
