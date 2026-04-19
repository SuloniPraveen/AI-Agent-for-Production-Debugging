"""Request-scoped log vector search: restrict chunks to the user's upload(s)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional
from uuid import UUID

# Tuple of batch UUID strings; empty tuple = user has no completed batch to search.
# Set for the duration of agent.get_response / get_stream_response.
_log_search_batch_ids: ContextVar[Optional[tuple[str, ...]]] = ContextVar(
    "log_search_batch_ids", default=None
)


def set_log_search_batch_ids(batch_ids: tuple[str, ...]) -> object:
    """Return a token for reset()."""
    return _log_search_batch_ids.set(batch_ids)


def reset_log_search_batch_ids(token: object) -> None:
    _log_search_batch_ids.reset(token)


def get_log_search_batch_ids() -> Optional[tuple[str, ...]]:
    """Resolved scope for this chat request, or None if not running inside chat."""
    return _log_search_batch_ids.get()
