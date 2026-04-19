"""Structured log RAG tool (pgvector + JSONB filters)."""

from typing import Optional

from langchain_core.tools import tool

from app.services.log_search import log_search_service


@tool
async def search_logs(
    query: str,
    service: Optional[str] = None,
    level: Optional[str] = None,
    time_from_iso: Optional[str] = None,
    time_to_iso: Optional[str] = None,
    top_k: int = 3,
) -> str:
    """Search ingested log chunks (vector similarity + optional metadata filters).

    Search is scoped to the user's current upload (latest completed batch, or the batch
    selected for this chat). Returns the top matching chunks by embedding similarity.

    Always prefer this tool for questions about production logs, errors, or services
    when logs may have been uploaded. Results include chunk_id — cite them in your answer.

    Args:
        query: Natural language or keyword search over log content.
        service: Filter meta.service (exact match, case-sensitive).
        level: Filter log level (e.g. ERROR, WARN); case-insensitive.
        time_from_iso: Lower bound on meta.timestamp (ISO-8601, e.g. 2025-01-01T00:00:00Z).
        time_to_iso: Upper bound on meta.timestamp (ISO-8601).
        top_k: Max chunks to return (default 3; server caps at LOG_SEARCH_MAX_TOP_K).

    Returns:
        JSON string with keys: citations (list with chunk_id, batch_id, timestamp, snippet, ...),
        context (text passages for reasoning). If no rows, context explains empty index.
    """
    return await log_search_service.search_json(
        query,
        service=service,
        level=level,
        time_from_iso=time_from_iso,
        time_to_iso=time_to_iso,
        top_k=top_k,
    )
