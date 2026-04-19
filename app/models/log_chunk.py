"""Vector-indexed log lines (Phase 1 log RAG)."""

from typing import Any, Optional
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

from app.core.config import settings


class LogChunk(SQLModel, table=True):
    """Chunk of log lines with embedding and filterable metadata."""

    __tablename__ = "log_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: UUID = Field(foreign_key="log_batches.id", index=True)
    chunk_index: int = Field(default=0, ge=0)
    content: str = Field(description="Contiguous log lines for this chunk.")
    embedding: Any = Field(sa_column=Column(Vector(settings.LOG_EMBEDDING_DIMENSIONS)))
    meta: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
        description="service, level, timestamp (ISO), line_start, line_end, etc.",
    )
