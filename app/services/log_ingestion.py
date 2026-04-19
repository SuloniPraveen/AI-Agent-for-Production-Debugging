"""Background log file ingestion: line/window chunking, OpenAI embeddings, log_chunk rows."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
)
from uuid import UUID

from openai import (
    OpenAI,
    RateLimitError,
)
from sqlmodel import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.models.log_batch import LogBatch, LogBatchStatus
from app.models.log_chunk import LogChunk
from app.services.database import database_service

_ISO_PREFIX = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)

# Plain-text lines: "<ISO> <service> <LEVEL> ..." (e.g. sample_service.log)
_PLAIN_LEVELS = frozenset(
    {"INFO", "WARN", "WARNING", "ERROR", "DEBUG", "TRACE", "FATAL", "CRITICAL", "NOTICE"}
)


def _parse_json_line(line: str) -> Optional[dict]:
    s = line.strip()
    if not s or not s.startswith("{"):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _extract_from_obj(obj: dict) -> dict:
    """Pull common log fields from a parsed JSON object (flat or shallow nested)."""
    out: dict = {}
    if not isinstance(obj, dict):
        return out
    for key in ("service", "service_name", "serviceName", "app", "application"):
        v = obj.get(key)
        if v and isinstance(v, str):
            out["service"] = v
            break
    for key in ("level", "severity", "log_level", "levelname"):
        v = obj.get(key)
        if v and isinstance(v, str):
            out["level"] = v.upper()[:32]
            break
    for key in ("timestamp", "time", "@timestamp", "ts", "date"):
        v = obj.get(key)
        if v is not None:
            out["timestamp"] = str(v)[:64]
            break
    k8s = obj.get("kubernetes")
    if isinstance(k8s, dict) and not out.get("service"):
        pod = k8s.get("pod_name") or k8s.get("podName")
        if pod:
            out["service"] = str(pod)[:256]
    return out


def _line_timestamp(line: str) -> Optional[str]:
    m = _ISO_PREFIX.match(line.strip())
    return m.group(1) if m else None


def _plain_service_level(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse `timestamp service LEVEL ...` when the line is not JSON."""
    s = line.strip()
    m = _ISO_PREFIX.match(s)
    if not m:
        return None, None
    rest = s[m.end() :].strip()
    parts = rest.split(None, 2)
    if len(parts) < 2:
        return None, None
    svc, lev = parts[0], parts[1]
    u = lev.upper()
    if u == "WARNING":
        u = "WARN"
    if u not in _PLAIN_LEVELS:
        return None, None
    return svc[:256], u[:32]


def _chunk_meta(start_idx: int, end_idx: int, text_block: str, raw_lines: List[str]) -> dict:
    """1-based line numbers; JSON lines populate meta; plain-text fills service/level tags."""
    meta: dict = {
        "line_start": start_idx + 1,
        "line_end": end_idx + 1,
    }
    slice_lines = raw_lines[start_idx : end_idx + 1]
    service_tags: set[str] = set()
    level_tags: set[str] = set()
    first_plain_service: Optional[str] = None
    first_plain_level: Optional[str] = None

    for ln in slice_lines:
        parsed = _parse_json_line(ln)
        if parsed:
            extracted = _extract_from_obj(parsed)
            if extracted:
                meta.update(extracted)
                if extracted.get("service"):
                    service_tags.add(str(extracted["service"])[:256])
                if extracted.get("level"):
                    level_tags.add(str(extracted["level"]).upper()[:32])
                break

        ts = _line_timestamp(ln)
        if ts and "timestamp" not in meta:
            meta["timestamp"] = ts

        svc, lvl = _plain_service_level(ln)
        if svc:
            service_tags.add(svc)
            if first_plain_service is None:
                first_plain_service = svc
        if lvl:
            level_tags.add(lvl)
            if first_plain_level is None:
                first_plain_level = lvl

    if service_tags:
        meta["service_tags"] = sorted(service_tags)
    if level_tags:
        meta["level_tags"] = sorted(level_tags)
    if first_plain_service is not None and "service" not in meta:
        meta["service"] = first_plain_service
    if first_plain_level is not None and "level" not in meta:
        meta["level"] = first_plain_level

    return meta


def _build_chunks(raw_lines: List[str]) -> List[Tuple[int, int, str, dict]]:
    """Return list of (start_idx, end_idx, content, meta).

    Windows grow line-by-line until ``settings.LOG_LINES_PER_CHUNK`` lines **or**
    ``settings.LOG_CHUNK_MAX_CHARS`` characters (plus newlines) — whichever is hit first — then flush.
    """
    max_lines = settings.LOG_LINES_PER_CHUNK
    max_chars = settings.LOG_CHUNK_MAX_CHARS
    chunks: List[Tuple[int, int, str, dict]] = []
    buf: List[str] = []
    char_count = 0
    start_idx = 0
    for i, line in enumerate(raw_lines):
        stripped = line.rstrip("\n\r")
        if not buf:
            start_idx = i
        buf.append(stripped)
        char_count += len(stripped) + 1
        if len(buf) >= max_lines or char_count >= max_chars:
            text = "\n".join(buf)
            meta = _chunk_meta(start_idx, i, text, raw_lines)
            chunks.append((start_idx, i, text, meta))
            buf = []
            char_count = 0
    if buf:
        end_idx = len(raw_lines) - 1
        text = "\n".join(buf)
        meta = _chunk_meta(start_idx, end_idx, text, raw_lines)
        chunks.append((start_idx, end_idx, text, meta))
    return chunks


def _chunk_specs_for_embed_api(
    chunks_spec: List[Tuple[int, int, str, dict]],
    max_chunks: int,
    max_chars: int,
) -> List[List[Tuple[int, int, str, dict]]]:
    """Split chunk list so each sub-batch stays under OpenAI input size limits."""
    batches: List[List[Tuple[int, int, str, dict]]] = []
    cur: List[Tuple[int, int, str, dict]] = []
    cur_chars = 0
    for spec in chunks_spec:
        _s, _e, text, _m = spec
        tlen = len(text)
        if cur and (
            len(cur) >= max_chunks or cur_chars + tlen > max_chars or tlen > max_chars
        ):
            batches.append(cur)
            cur = []
            cur_chars = 0
        # Single huge chunk: still send alone (may fail; caller should tune LOG_CHUNK_MAX_CHARS)
        cur.append(spec)
        cur_chars += tlen
    if cur:
        batches.append(cur)
    return batches


@retry(
    reraise=True,
    stop=stop_after_attempt(24),
    wait=wait_exponential(multiplier=1, min=2, max=90),
    retry=retry_if_exception_type(RateLimitError),
)
def _embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    kwargs = {"model": settings.LOG_EMBEDDING_MODEL, "input": texts}
    if settings.LOG_EMBEDDING_MODEL.startswith("text-embedding-3") and settings.LOG_EMBEDDING_DIMENSIONS != 1536:
        kwargs["dimensions"] = settings.LOG_EMBEDDING_DIMENSIONS
    resp = client.embeddings.create(**kwargs)
    by_idx = {d.index: list(d.embedding) for d in resp.data}
    return [by_idx[j] for j in range(len(texts))]


def run_ingestion_sync(batch_id: UUID, file_path: str) -> None:
    """Sync pipeline: read file, chunk, embed, insert. Runs inside thread pool."""
    engine = database_service.engine
    path = Path(file_path)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _fail_batch(engine, batch_id, str(e))
        return

    lines = raw.splitlines(keepends=True)
    if not lines:
        _fail_batch(engine, batch_id, "empty file")
        return

    chunks_spec = _build_chunks(lines)
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    max_chunks = max(1, min(settings.LOG_INGEST_EMBED_BATCH, 256))
    max_chars = max(10_000, settings.LOG_INGEST_EMBED_MAX_CHARS)
    embed_batches = _chunk_specs_for_embed_api(chunks_spec, max_chunks, max_chars)

    with Session(engine) as session:
        batch = session.get(LogBatch, batch_id)
        if batch is None:
            logger.error("log_batch_missing", batch_id=str(batch_id))
            return
        batch.status = LogBatchStatus.PROCESSING.value
        batch.line_count = len(lines)
        session.add(batch)
        session.commit()

    all_rows: List[LogChunk] = []
    chunk_index = 0
    pause = max(0.0, settings.LOG_INGEST_EMBED_PAUSE_SECONDS)
    n_batches = len(embed_batches)
    for bi, batch_specs in enumerate(embed_batches):
        texts = [t[2] for t in batch_specs]
        try:
            embeddings = _embed_batch(client, texts)
        except Exception as e:
            logger.exception("log_embed_batch_failed", error=str(e))
            _fail_batch(engine, batch_id, f"embedding failed: {e}")
            return
        for j, (_start, _end, text, meta) in enumerate(batch_specs):
            emb = embeddings[j]
            if len(emb) != settings.LOG_EMBEDDING_DIMENSIONS:
                logger.warning(
                    "log_embedding_dim",
                    expected=settings.LOG_EMBEDDING_DIMENSIONS,
                    got=len(emb),
                )
            all_rows.append(
                LogChunk(
                    batch_id=batch_id,
                    chunk_index=chunk_index,
                    content=text[:50000],
                    embedding=emb,
                    meta=meta,
                )
            )
            chunk_index += 1

        if pause > 0 and bi + 1 < n_batches:
            time.sleep(pause)

    try:
        with Session(engine) as session:
            for row in all_rows:
                session.add(row)
            batch = session.get(LogBatch, batch_id)
            if batch:
                batch.status = LogBatchStatus.COMPLETED.value
                batch.chunk_count = len(all_rows)
                session.add(batch)
            session.commit()
        logger.info(
            "log_ingest_completed",
            batch_id=str(batch_id),
            chunks=len(all_rows),
            lines=len(lines),
        )
    except Exception as e:
        logger.exception("log_ingest_commit_failed", error=str(e))
        _fail_batch(engine, batch_id, str(e))


def _fail_batch(engine, batch_id: UUID, message: str) -> None:
    with Session(engine) as session:
        batch = session.get(LogBatch, batch_id)
        if batch:
            batch.status = LogBatchStatus.FAILED.value
            batch.error_message = message[:4000]
            session.add(batch)
            session.commit()
    logger.error("log_ingest_failed", batch_id=str(batch_id), error=message)


def create_pending_batch(user_id: int, filename: str) -> LogBatch:
    with Session(database_service.engine) as session:
        batch = LogBatch(user_id=user_id, filename=filename[:512], status=LogBatchStatus.PENDING.value)
        session.add(batch)
        session.commit()
        session.refresh(batch)
        return batch


def get_batch_for_user(batch_id: UUID, user_id: int) -> Optional[LogBatch]:
    with Session(database_service.engine) as session:
        b = session.get(LogBatch, batch_id)
        if b is None or b.user_id != user_id:
            return None
        return b
