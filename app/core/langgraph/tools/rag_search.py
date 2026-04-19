"""Vector RAG tool: search ingested logs and runbooks."""

from langchain_core.tools import tool

from app.services.rag import rag_service


@tool
async def search_incident_knowledge(query: str, source_type: str = "any") -> str:
    """Search ingested production logs and runbooks (pgvector RAG).

    Use when the user asks about incidents, errors, services, or internal procedures
    that might appear in indexed logs or runbooks.

    Args:
        query: Focused search query (symptoms, error messages, service names).
        source_type: One of: 'log', 'runbook', or 'any' (search both).

    Returns:
        Top matching passages with source labels, or guidance if the index is empty.
    """
    return await rag_service.search(query, source_type=source_type)
