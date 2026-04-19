"""Stable cache keys for log vector search.

We key by canonical query + filters + embedding model/dim. For a fixed OpenAI embedding
model, identical text yields identical vectors, so this matches caching
``hash(embed(query)) + filters`` for repeat traffic while allowing a cache hit **without**
calling embed or Postgres on the hot path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from app.core.config import settings


def log_search_cache_key(
    query: str,
    service: Optional[str],
    level: Optional[str],
    time_from_iso: Optional[str],
    time_to_iso: Optional[str],
    top_k: int,
    batch_scope: Optional[tuple[str, ...]] = None,
) -> str:
    payload = {
        "q": query.strip().lower(),
        "service": service or "",
        "level": (level or "").lower(),
        "t0": time_from_iso or "",
        "t1": time_to_iso or "",
        "k": int(top_k),
        "batches": list(batch_scope) if batch_scope else [],
        "model": settings.LOG_EMBEDDING_MODEL,
        "dim": int(settings.LOG_EMBEDDING_DIMENSIONS),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"logsearch:v3:{digest}"
