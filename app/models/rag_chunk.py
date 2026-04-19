"""Vector-indexed chunks for log and runbook RAG."""

from typing import (
    Any,
    Optional,
)

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

from app.core.config import settings


class RagChunk(SQLModel, table=True):
    """One searchable text chunk with an embedding (pgvector)."""

    __tablename__ = "rag_chunk"

    id: Optional[int] = Field(default=None, primary_key=True)
    doc_type: str = Field(
        max_length=32,
        index=True,
        description="Discriminator: typically 'log' or 'runbook'.",
    )
    source_label: str = Field(default="", max_length=512, description="File name, service, or doc id.")
    content: str = Field(description="Chunk text used for retrieval and display.")
    embedding: Any = Field(sa_column=Column(Vector(settings.RAG_EMBEDDING_DIMENSIONS)))
