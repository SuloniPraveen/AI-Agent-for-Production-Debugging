"""Cosine similarity search over log_chunk with JSONB metadata filters + optional Redis cache."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import (
    Any,
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
from app.core.metrics import (
    cache_hits_total,
    cache_misses_total,
    retrieval_latency_seconds,
)
from app.services.database import database_service
from app.services.log_search_cache import log_search_cache_key
from app.services.log_search_scope import get_log_search_batch_ids
from app.services.redis_client import get_redis

CACHE_LABEL = "log_search"


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in embedding) + "]"


def _sync_search(
    engine,
    query_embedding: List[float],
    batch_ids: List[str],
    service: Optional[str],
    level: Optional[str],
    time_from_iso: Optional[str],
    time_to_iso: Optional[str],
    limit: int,
) -> List[Tuple[int, str, str, dict, float]]:
    """Rows: id, batch_id str, content, meta dict, distance."""
    vec = _vector_literal(query_embedding)
    lim = max(1, min(limit, settings.LOG_SEARCH_MAX_TOP_K))
    conditions: List[str] = ["1=1"]
    params: dict[str, Any] = {"qv": vec, "lim": lim}

    if batch_ids:
        placeholders = ", ".join(f":bid{i}" for i in range(len(batch_ids)))
        conditions.append(f"batch_id IN ({placeholders})")
        for i, bid in enumerate(batch_ids):
            params[f"bid{i}"] = bid

    # Avoid empty-string ::timestamptz (invalid) and skip obvious non-ISO garbage before cast.
    ts_trim = "NULLIF(btrim(meta->>'timestamp'), '')"
    ts_shape = (
        f"({ts_trim} IS NOT NULL AND {ts_trim} ~ "
        f"'^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}[T ][0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}')"
    )

    if service:
        # jsonb_build_array(:param) is fragile with some drivers; use a single jsonb bind.
        conditions.append(
            "(meta->>'service' = :svc_eq OR "
            "(COALESCE(meta->'service_tags','[]'::jsonb) @> CAST(:svc_tag_json AS jsonb)))"
        )
        params["svc_eq"] = service
        params["svc_tag_json"] = json.dumps([service])
    if level:
        conditions.append(
            "(LOWER(COALESCE(meta->>'level','')) = LOWER(:lvl) OR EXISTS ("
            "SELECT 1 FROM jsonb_array_elements_text(COALESCE(meta->'level_tags','[]'::jsonb)) AS elt "
            "WHERE LOWER(elt) = LOWER(:lvl)))"
        )
        params["lvl"] = level
    if time_from_iso or time_to_iso:
        conditions.append(ts_shape)
        # Never use ":name::timestamptz" — SQLAlchemy text() treats ":" as bind syntax and the
        # second "::" breaks parsing (surfaces as driver/SQL "syntax" errors). Use CAST instead.
        if time_from_iso:
            conditions.append(
                f"CAST(({ts_trim}) AS timestamptz) >= CAST(:tfrom AS timestamptz)"
            )
            params["tfrom"] = time_from_iso
        if time_to_iso:
            conditions.append(
                f"CAST(({ts_trim}) AS timestamptz) <= CAST(:tto AS timestamptz)"
            )
            params["tto"] = time_to_iso

    where_sql = " AND ".join(conditions)
    stmt = text(
        f"""
        SELECT id, batch_id::text, content, meta,
               (embedding <=> CAST(:qv AS vector)) AS dist
        FROM log_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> CAST(:qv AS vector)
        LIMIT :lim
        """
    )

    with Session(engine) as session:
        result = session.execute(stmt, params)
        rows = []
        for r in result.fetchall():
            meta = r[3] if isinstance(r[3], dict) else {}
            rows.append((int(r[0]), str(r[1]), str(r[2]), meta, float(r[4])))
        return rows


def _snippet(text: str, max_len: int = 480) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _rows_to_payload(
    rows: List[Tuple[int, str, str, dict, float]],
) -> dict:
    citations: List[dict] = []
    context_parts: List[str] = []
    for cid, bid, content, meta, dist in rows:
        snip = _snippet(content)
        citations.append(
            {
                "chunk_id": cid,
                "batch_id": bid,
                "timestamp": meta.get("timestamp"),
                "snippet": snip,
                "service": meta.get("service"),
                "level": meta.get("level"),
                "line_start": meta.get("line_start"),
                "line_end": meta.get("line_end"),
            }
        )
        context_parts.append(
            f"[chunk_id={cid} batch={bid} dist={dist:.4f} service={meta.get('service')} "
            f"level={meta.get('level')} ts={meta.get('timestamp')} lines={meta.get('line_start')}-{meta.get('line_end')}]\n{content}"
        )
    payload: dict = {
        "citations": citations,
        "context": "\n\n---\n\n".join(context_parts) if context_parts else "",
    }
    if not rows:
        payload["context"] = (
            "No log chunks matched in the selected upload(s). Try a broader query, relax time/service filters, "
            "or confirm ingestion completed for this file."
        )
    return payload


class LogSearchService:
    """Async facade over pgvector + metadata filters + Redis cache."""

    def __init__(self) -> None:
        self._engine = database_service.engine
        self._model = settings.LOG_EMBEDDING_MODEL

    async def embed_query(self, query: str) -> List[float]:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        kwargs = {"model": self._model, "input": query}
        if settings.LOG_EMBEDDING_MODEL.startswith("text-embedding-3") and settings.LOG_EMBEDDING_DIMENSIONS != 1536:
            kwargs["dimensions"] = settings.LOG_EMBEDDING_DIMENSIONS
        resp = await client.embeddings.create(**kwargs)
        return list(resp.data[0].embedding)

    async def search_json(
        self,
        query: str,
        service: Optional[str] = None,
        level: Optional[str] = None,
        time_from_iso: Optional[str] = None,
        time_to_iso: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Return JSON string for ToolMessage: citations + context for the LLM."""
        t_wall = time.perf_counter()
        scope = get_log_search_batch_ids()
        if scope is None:
            return json.dumps(
                {
                    "citations": [],
                    "context": "Log search is only available during an authenticated chat request.",
                    "error": "log_search_scope_missing",
                }
            )
        if len(scope) == 0:
            return json.dumps(
                {
                    "citations": [],
                    "context": (
                        "No completed log upload found for this account. Upload a log file, wait until "
                        "ingestion status is completed, then ask again."
                    ),
                    "error": "no_log_batch_scope",
                }
            )

        batch_list = list(scope)
        tk = int(top_k) if top_k else settings.LOG_SEARCH_DEFAULT_TOP_K
        tk = max(1, min(tk, settings.LOG_SEARCH_MAX_TOP_K))
        batch_scope_key: tuple[str, ...] = tuple(sorted(batch_list))

        for label, val in (("time_from_iso", time_from_iso), ("time_to_iso", time_to_iso)):
            if val:
                try:
                    datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    return json.dumps(
                        {
                            "error": f"invalid {label}; use ISO-8601",
                            "citations": [],
                            "context": "",
                        }
                    )

        cache_key = log_search_cache_key(
            query, service, level, time_from_iso, time_to_iso, tk, batch_scope=batch_scope_key
        )
        redis = await get_redis()
        if redis is not None:
            try:
                cached = await redis.get(cache_key)
            except Exception as e:
                logger.warning("redis_get_failed", error=str(e))
                cached = None
            if cached:
                cache_hits_total.labels(cache=CACHE_LABEL).inc()
                elapsed = time.perf_counter() - t_wall
                retrieval_latency_seconds.labels(phase="total_hit").observe(elapsed)
                return cached

        cache_misses_total.labels(cache=CACHE_LABEL).inc()

        try:
            t0 = time.perf_counter()
            qvec = await self.embed_query(query)
            retrieval_latency_seconds.labels(phase="embed").observe(time.perf_counter() - t0)
        except Exception as e:
            logger.exception("log_search_embed_failed", error=str(e))
            return json.dumps({"error": str(e), "citations": [], "context": ""})

        try:
            t1 = time.perf_counter()
            rows = await asyncio.to_thread(
                _sync_search,
                self._engine,
                qvec,
                batch_list,
                service,
                level,
                time_from_iso,
                time_to_iso,
                tk,
            )
            retrieval_latency_seconds.labels(phase="vector_search").observe(time.perf_counter() - t1)
        except Exception as e:
            logger.exception("log_search_sql_failed", error=str(e))
            return json.dumps({"error": str(e), "citations": [], "context": ""})

        payload = _rows_to_payload(rows)
        out = json.dumps(payload, default=str)

        if redis is not None and "error" not in payload:
            try:
                await redis.set(cache_key, out, ex=settings.LOG_SEARCH_CACHE_TTL_SECONDS)
            except Exception as e:
                logger.warning("redis_set_failed", error=str(e))

        elapsed_total = time.perf_counter() - t_wall
        retrieval_latency_seconds.labels(phase="total_miss").observe(elapsed_total)
        return out


log_search_service = LogSearchService()
