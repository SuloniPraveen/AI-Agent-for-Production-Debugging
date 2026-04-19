"""Embedding and vector similarity search over ingested log/runbook chunks."""

import asyncio
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
)

from openai import AsyncOpenAI
from sqlalchemy import text
from sqlmodel import Session

from app.core.config import settings
from app.core.logging import logger
from app.models.rag_chunk import RagChunk
from app.services.database import database_service


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"


def _sync_search(
    engine,
    query_embedding: List[float],
    doc_type: Optional[str],
    limit: int,
) -> List[Tuple[str, str, str, float]]:
    """Return rows: (doc_type, source_label, content, distance)."""
    vec = _vector_literal(query_embedding)
    lim = max(1, min(limit, 20))
    with Session(engine) as session:
        if doc_type in ("log", "runbook"):
            stmt = text(
                """
                SELECT doc_type, source_label, content,
                       (embedding <=> CAST(:qv AS vector)) AS dist
                FROM rag_chunk
                WHERE doc_type = :dt
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :lim
                """
            )
            result = session.execute(stmt, {"qv": vec, "dt": doc_type, "lim": lim})
        else:
            stmt = text(
                """
                SELECT doc_type, source_label, content,
                       (embedding <=> CAST(:qv AS vector)) AS dist
                FROM rag_chunk
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :lim
                """
            )
            result = session.execute(stmt, {"qv": vec, "lim": lim})
        return [(r[0], r[1], r[2], float(r[3])) for r in result.fetchall()]


def _sync_insert_chunks(engine, chunks: List[RagChunk]) -> int:
    with Session(engine) as session:
        for c in chunks:
            session.add(c)
        session.commit()
    return len(chunks)


class RagService:
    """OpenAI embeddings + pgvector cosine search."""

    def __init__(self) -> None:
        self._engine = database_service.engine
        self._model = settings.RAG_EMBEDDING_MODEL

    async def embed_query(self, query: str) -> List[float]:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        kwargs = {"model": self._model, "input": query}
        if settings.RAG_EMBEDDING_MODEL.startswith("text-embedding-3") and settings.RAG_EMBEDDING_DIMENSIONS != 1536:
            kwargs["dimensions"] = settings.RAG_EMBEDDING_DIMENSIONS
        resp = await client.embeddings.create(**kwargs)
        vec = list(resp.data[0].embedding)
        if len(vec) != settings.RAG_EMBEDDING_DIMENSIONS:
            logger.warning(
                "rag_embedding_dimension_mismatch",
                expected=settings.RAG_EMBEDDING_DIMENSIONS,
                got=len(vec),
            )
        return vec

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed (one API call when possible)."""
        if not texts:
            return []
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        kwargs = {"model": self._model, "input": texts}
        if settings.RAG_EMBEDDING_MODEL.startswith("text-embedding-3") and settings.RAG_EMBEDDING_DIMENSIONS != 1536:
            kwargs["dimensions"] = settings.RAG_EMBEDDING_DIMENSIONS
        resp = await client.embeddings.create(**kwargs)
        by_index = {item.index: list(item.embedding) for item in resp.data}
        return [by_index[i] for i in range(len(texts))]

    async def search(self, query: str, source_type: str = "any", top_k: Optional[int] = None) -> str:
        """Run similarity search and return a single string for the LLM tool."""
        top_k = top_k or settings.RAG_TOP_K
        st = (source_type or "any").strip().lower()
        doc_filter: Optional[str] = None
        if st in ("log", "runbook"):
            doc_filter = st
        elif st != "any":
            doc_filter = None

        try:
            qvec = await self.embed_query(query)
        except Exception as e:
            logger.exception("rag_embed_query_failed", error=str(e))
            return f"Embedding failed: {e!s}"

        try:
            rows = await asyncio.to_thread(_sync_search, self._engine, qvec, doc_filter, top_k)
        except Exception as e:
            logger.exception("rag_search_failed", error=str(e))
            return f"Search failed: {e!s}"

        if not rows:
            return (
                "No matching ingested documents. Ingest logs or runbooks with "
                "`uv run python scripts/ingest_rag_documents.py` (see README)."
            )

        parts: List[str] = []
        for i, (dt, src, content, dist) in enumerate(rows, start=1):
            parts.append(f"--- Result {i} (type={dt}, source={src}, distance={dist:.4f}) ---\n{content}")
        return "\n\n".join(parts)

    async def insert_chunks(self, chunks: List[RagChunk]) -> int:
        return await asyncio.to_thread(_sync_insert_chunks, self._engine, chunks)


rag_service = RagService()
